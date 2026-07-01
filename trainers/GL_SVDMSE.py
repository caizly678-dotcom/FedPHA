import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from trainers.spf_utils import compute_shared_basis, project_to_basis
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'GL_SVDMSE',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.GL_SVDMSE.N_CTX
        ctx_init = cfg.TRAINER.GL_SVDMSE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.N = cfg.TRAINER.GL_SVDMSE.N
        self.ratio = cfg.TRAINER.GL_SVDMSE.ratio
        self.use_spf = cfg.TRAINER.GL_SVDMSE.USE_SPF
        self.spf_energy = cfg.TRAINER.GL_SVDMSE.SPF_ENERGY
        self.spf_min_rank = cfg.TRAINER.GL_SVDMSE.SPF_MIN_RANK
        self.spf_max_rank = cfg.TRAINER.GL_SVDMSE.SPF_MAX_RANK
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
            ctx_global = ctx_vectors.unsqueeze(0).repeat(self.N, 1, 1)
            ctx_local = ctx_vectors.unsqueeze(0).repeat(self.N, 1, 1)

        else:
            # random initialization
            if cfg.TRAINER.GL_SVDMSE.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                # ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype) 
                ctx_global = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
                ctx_local = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype) 
            
            # nn.init.normal_(ctx_vectors, std=0.02)   # define the prompt to be trained
            nn.init.normal_(ctx_global, std=0.02)   # define the prompt to be trained
            nn.init.normal_(ctx_local, std=0.02)   # define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)    

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_global = nn.Parameter(ctx_global)
        self.ctx_local = nn.Parameter(ctx_local)
        
        classnames = [name.replace("_", " ") for name in classnames]   
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])   
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1) 

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.GL_SVDMSE.CLASS_TOKEN_POSITION

    def _build_prompts_from_ctx(self, ctx):
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        ctx = ctx.permute(1, 0, 2, 3).contiguous().view(
            self.N * self.n_cls, self.n_ctx, ctx.shape[-1]
        )
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

    def get_spf_aux(self, ctx_local, ctx_global):
        with torch.no_grad():
            basis, _ = compute_shared_basis(
                ctx_global.detach(),
                energy=self.spf_energy,
                min_rank=self.spf_min_rank,
                max_rank=self.spf_max_rank,
            )
        basis = basis.detach()

        local_shared = project_to_basis(ctx_local, basis)

        with torch.no_grad():
            global_shared = project_to_basis(ctx_global.detach(), basis)

        shared_pull_loss = F.mse_loss(local_shared.float(), global_shared.float())

        aux = {
            "shared_pull_loss": shared_pull_loss,
            "svd_rank": torch.tensor(float(basis.shape[1]), device=ctx_local.device),
        }
        return aux

    def compute_null_space(self, global_ctx, ratio=0.8):
        global_ctx = global_ctx.view(-1, global_ctx.shape[-1])  # Flatten: (N * n_ctx, ctx_dim)
        global_ctx = global_ctx.to(torch.float32)
        
        try:
            U, S, V = torch.svd(global_ctx)           
            # U = [len, len]
            # S = [len]
            # V = [dim, dim]
        except RuntimeError as e:
            print(f"SVD failed on GPU: {e}")
            global_ctx_cpu = global_ctx.cpu()
            U, S, V = torch.svd(global_ctx_cpu)
            V = V.to(global_ctx.device)

        cutoff = int(S.shape[0] * (1 - ratio))
        V2 = V[:, cutoff:]

        return V2.to(global_ctx.dtype)

    def forward(self):
        if self.use_spf:
            if self.class_token_position != "end":
                raise NotImplementedError("SPF-FedPHA only supports CLASS_TOKEN_POSITION == 'end'")

            prompts_global = self._build_prompts_from_ctx(self.ctx_global)
            prompts_local = self._build_prompts_from_ctx(self.ctx_local)
            aux = self.get_spf_aux(self.ctx_local, self.ctx_global)
            return prompts_global, prompts_local, aux

        ctx = self.ctx_local

        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        ctx = ctx.permute(1, 0, 2, 3).contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[-1])
        
        ctx_global = self.ctx_global
        null_space = self.compute_null_space(ctx_global, self.ratio)  
        
        ctx_global = ctx_global.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        ctx_global = ctx_global.permute(1, 0, 2, 3).contiguous().view(self.N * self.n_cls, self.n_ctx, ctx_global.shape[-1])

        ctx_flat = self.ctx_local.view(-1, self.ctx_local.shape[-1])  # Flatten [ctx, 512]
        null_space = null_space.to(ctx_flat.dtype)

        projected_ctx = torch.mm(ctx_flat, torch.mm(null_space, null_space.T))
        projected_ctx_local = projected_ctx.view(self.ctx_local.shape)
        projected_ctx_local = projected_ctx_local.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        projected_ctx_local = projected_ctx_local.permute(1, 0, 2, 3).contiguous().view(self.N * self.n_cls, self.n_ctx, ctx_global.shape[-1])
        
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_global = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx_global,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_projected_local = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    projected_ctx_local,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, prompts_global, prompts_projected_local


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.N = cfg.TRAINER.GL_SVDMSE.N
        self.spf_late_alpha = float(cfg.TRAINER.GL_SVDMSE.SPF_LATE_ALPHA)

    def forward(self, image, idx=None, return_all=False):
        tokenized_prompts = self.tokenized_prompts
        out = self.prompt_learner()

        if self.prompt_learner.use_spf:
            prompts_global, prompts_local, aux = out
            text_features_global = self.text_encoder(prompts_global, tokenized_prompts)
            text_features_global = text_features_global / text_features_global.norm(dim=-1, keepdim=True)
            text_features_local = self.text_encoder(prompts_local, tokenized_prompts)
            text_features_local = text_features_local / text_features_local.norm(dim=-1, keepdim=True)
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits_global = logit_scale * image_features @ text_features_global.t()
            logits_local = logit_scale * image_features @ text_features_local.t()
            alpha = self.spf_late_alpha
            logits_fused = alpha * logits_global + (1.0 - alpha) * logits_local

            if self.training:
                return logits_fused, logits_global, logits_local, aux

            if return_all:
                return logits_fused, logits_global, logits_local

            return logits_fused

        prompts, prompts_global, prompts_projected_local = out
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if self.training == True:
            text_features_global = self.text_encoder(prompts_global, tokenized_prompts)
            text_features_global = text_features_global / text_features_global.norm(dim=-1, keepdim=True)
            text_features_projected_local = self.text_encoder(prompts_projected_local, tokenized_prompts)
            text_features_projected_local = text_features_projected_local / text_features_projected_local.norm(dim=-1, keepdim=True)
            logits_global = logit_scale * image_features @ text_features_global.t()
            return logits, text_features_global, text_features, text_features_projected_local, logits_global

        return logits


# @TRAINER_REGISTRY.register()
class GL_SVDMSE(TrainerX):
    """
    It is based on CoOp.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.GL_SVDMSE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        self.lambda_orthogonal = cfg.TRAINER.GL_SVDMSE.lambda_orthogonal
        self.alpha = cfg.TRAINER.GL_SVDMSE.alpha
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.GL_SVDMSE.PREC == "fp32" or cfg.TRAINER.GL_SVDMSE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()   

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        if cfg.DATASET.NAME== "ImageNet":
            self.device =  torch.device("cuda:0")
            # device0 = torch.device("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
            self.model.text_encoder=nn.DataParallel(self.model.text_encoder)
        else:
            self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.GL_SVDMSE.PREC == "amp" else None


    def forward_backward(self, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.GL_SVDMSE.PREC

        if self.model.prompt_learner.use_spf:
            lambda_fused = self.cfg.TRAINER.GL_SVDMSE.SPF_FUSED_CE_LAMBDA
            lambda_global = self.cfg.TRAINER.GL_SVDMSE.SPF_GLOBAL_CE_LAMBDA
            lambda_local = self.cfg.TRAINER.GL_SVDMSE.SPF_LOCAL_CE_LAMBDA
            lambda_sum = lambda_fused + lambda_global + lambda_local

            if prec == "amp":
                with autocast():
                    output, output_global, output_local, aux = self.model(image)
                    loss_fused = F.cross_entropy(output, label)
                    loss_global = F.cross_entropy(output_global, label)
                    loss_local = F.cross_entropy(output_local, label)
                    loss_cls = (
                        lambda_fused * loss_fused
                        + lambda_global * loss_global
                        + lambda_local * loss_local
                    ) / lambda_sum
                    loss = (
                        loss_cls
                        + self.cfg.TRAINER.GL_SVDMSE.SPF_SHARED_LAMBDA
                        * aux["shared_pull_loss"]
                    )
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output, output_global, output_local, aux = self.model(image)
                loss_fused = F.cross_entropy(output, label)
                loss_global = F.cross_entropy(output_global, label)
                loss_local = F.cross_entropy(output_local, label)
                loss_cls = (
                    lambda_fused * loss_fused
                    + lambda_global * loss_global
                    + lambda_local * loss_local
                ) / lambda_sum
                loss = (
                    loss_cls
                    + self.cfg.TRAINER.GL_SVDMSE.SPF_SHARED_LAMBDA
                    * aux["shared_pull_loss"]
                )
                self.model_backward_and_update(loss)

            loss_summary = {
                "loss": loss.item(),
                "loss_fused": loss_fused.item(),
                "loss_global": loss_global.item(),
                "loss_local": loss_local.item(),
                "acc": compute_accuracy(output, label)[0].item(),
                "acc_global": compute_accuracy(output_global, label)[0].item(),
                "acc_local": compute_accuracy(output_local, label)[0].item(),
                "spf_rank": float(aux["svd_rank"].item()),
                "spf_shared_loss": float(aux["shared_pull_loss"].item()),
            }
        else:
            if prec == "amp":
                with autocast():
                    output = self.model(image)
                    loss = F.cross_entropy(output, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output, global_features, local_features, projected_local_features, output_global = self.model(image)

                pull_loss = F.mse_loss(local_features, projected_local_features)

                alpha = self.alpha
                push_loss = F.relu(alpha - torch.norm(local_features - global_features, dim=-1)).mean()
                lambda_pull = 1.0
                lambda_push = 1.0

                loss = F.cross_entropy(output, label)
                loss2 = F.cross_entropy(output_global, label)
                loss += loss2
                loss += lambda_pull * pull_loss + lambda_push * push_loss
                self.model_backward_and_update(loss)

            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    @torch.no_grad()
    def test(self, split=None, is_global=False, current_epoch=0, idx=-1, global_test=False):
        if not self.model.prompt_learner.use_spf:
            return super().test(
                split=split,
                is_global=is_global,
                current_epoch=current_epoch,
                idx=idx,
                global_test=global_test,
            )

        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.fed_test_loader_x_dict[idx]

        print(f"Evaluate on the client{idx}_{split} set")

        global_correct = 0
        local_correct = 0
        total = 0

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output, output_global, output_local = self.model(
                input,
                idx=idx,
                return_all=True,
            )
            self.evaluator.process(output, label)

            global_correct += int(output_global.max(1)[1].eq(label).sum().item())
            local_correct += int(output_local.max(1)[1].eq(label).sum().item())
            total += label.shape[0]

        results = self.evaluator.evaluate()
        self.evaluator.reset()

        fused_accuracy = results["accuracy"]
        global_accuracy = 100.0 * global_correct / total
        local_accuracy = 100.0 * local_correct / total

        results["fused_accuracy"] = fused_accuracy
        results["global_only_accuracy"] = global_accuracy
        results["local_only_accuracy"] = local_accuracy

        if not is_global and idx < 0:
            current_epoch = self.epoch
        for k, v in results.items():
            tag = f"{split}/{k}"
            if not is_global:
                tag = f"{tag}/{str(idx)}"
            self.write_scalar(tag, v, current_epoch)

        return results

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
