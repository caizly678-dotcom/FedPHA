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
        self.register_buffer(
            "fusion_gamma",
            torch.tensor([cfg.TRAINER.GL_SVDMSE.SPF_GAMMA_INIT], dtype=torch.float32)
        )
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

    def build_prompts_from_ctx(self, ctx):
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        ctx = ctx.permute(1, 0, 2, 3).contiguous().view(
            self.N * self.n_cls, self.n_ctx, ctx.shape[-1]
        )
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

    def fuse_ctx_spf(self, ctx_local, ctx_global, spf_scale=1.0):
        basis, _ = compute_shared_basis(
            ctx_global.detach(),
            energy=self.spf_energy,
            min_rank=self.spf_min_rank,
            max_rank=self.spf_max_rank,
        )
        local_shared = project_to_basis(ctx_local, basis)
        global_shared = project_to_basis(ctx_global, basis)

        scale = torch.as_tensor(
            spf_scale,
            device=ctx_local.device,
            dtype=ctx_local.dtype
        ).clamp(0.0, 1.0)
        gamma_base = self.fusion_gamma.to(device=ctx_local.device, dtype=ctx_local.dtype).view(1, 1, 1)
        gamma = gamma_base * scale.view(1, 1, 1)
        fused_ctx = ctx_local + gamma * (global_shared.detach() - local_shared)

        aux = {
            "gamma": gamma.detach().float().mean(),
            "gamma_base": gamma_base.detach().float().mean(),
            "spf_scale": scale.detach().float(),
            "svd_rank": torch.tensor(float(basis.shape[1]), device=ctx_local.device),
            "prompt_shared_mse": F.mse_loss(
                local_shared.detach().float(),
                global_shared.detach().float()
            ),
        }
        return fused_ctx, local_shared, global_shared, aux

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

    def forward(self, spf_scale=1.0):
        if self.use_spf:
            if self.class_token_position != "end":
                raise NotImplementedError("SPF-FedPHA only supports CLASS_TOKEN_POSITION == 'end'")

            fused_ctx, local_shared, global_shared, aux = self.fuse_ctx_spf(
                self.ctx_local, self.ctx_global, spf_scale=spf_scale
            )

            prompts = self.build_prompts_from_ctx(fused_ctx)
            prompts_global = self.build_prompts_from_ctx(self.ctx_global)
            prompts_local_shared = self.build_prompts_from_ctx(local_shared)
            prompts_global_shared = self.build_prompts_from_ctx(global_shared.detach())
            expected_shape = prompts.shape
            expected_dtype = prompts.dtype
            for name, prompt in [
                ("prompts_global", prompts_global),
                ("prompts_local_shared", prompts_local_shared),
                ("prompts_global_shared", prompts_global_shared),
            ]:
                assert prompt.shape == expected_shape, (
                    f"{name} shape {prompt.shape} != fused prompts shape {expected_shape}"
                )
                assert prompt.dtype == expected_dtype, (
                    f"{name} dtype {prompt.dtype} != fused prompts dtype {expected_dtype}"
                )
            prompts_local = self.build_prompts_from_ctx(self.ctx_local)
            return (
                prompts_local,
                prompts,
                prompts_global,
                prompts_local_shared,
                prompts_global_shared,
                aux,
            )

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

    def forward(self, image, idx=None, spf_scale=1.0):
        tokenized_prompts = self.tokenized_prompts
        out = self.prompt_learner(spf_scale=spf_scale)

        if self.prompt_learner.use_spf:
            (
                prompts_local,
                prompts_fused,
                prompts_global,
                prompts_local_shared,
                prompts_global_shared,
                aux,
            ) = out
            local_features = self.text_encoder(prompts_local, tokenized_prompts)
            local_features = local_features / local_features.norm(dim=-1, keepdim=True)
            fused_features = self.text_encoder(prompts_fused, tokenized_prompts)
            fused_features = fused_features / fused_features.norm(dim=-1, keepdim=True)
            global_features = self.text_encoder(prompts_global, tokenized_prompts)
            global_features = global_features / global_features.norm(dim=-1, keepdim=True)
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits_local = logit_scale * image_features @ local_features.t()
            logits_fused = logit_scale * image_features @ fused_features.t()
            logits_global = logit_scale * image_features @ global_features.t()

            if self.training:
                local_shared_features = self.text_encoder(
                    prompts_local_shared, tokenized_prompts
                )
                local_shared_features = local_shared_features / local_shared_features.norm(
                    dim=-1, keepdim=True
                )
                global_shared_features = self.text_encoder(
                    prompts_global_shared, tokenized_prompts
                )
                global_shared_features = global_shared_features / global_shared_features.norm(
                    dim=-1, keepdim=True
                )
                cosine_sim = torch.sum(
                    local_shared_features.float()
                    * global_shared_features.detach().float(),
                    dim=-1,
                ).mean()
                aux["shared_pull_loss"] = 1.0 - cosine_sim
                assert logits_local.shape == logits_fused.shape == logits_global.shape, (
                    "SPF logits shape mismatch: "
                    f"local={logits_local.shape}, "
                    f"fused={logits_fused.shape}, "
                    f"global={logits_global.shape}"
                )
                return logits_local, logits_fused, logits_global, aux

            return {
                "local": logits_local,
                "fused": logits_fused,
                "global": logits_global,
            }

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

    def get_spf_warmup_scale(self, global_epoch):
        warmup_rounds = int(self.cfg.TRAINER.GL_SVDMSE.SPF_WARMUP_ROUNDS)
        if warmup_rounds <= 0 or global_epoch < 0:
            return 1.0
        return min(1.0, float(global_epoch + 1) / float(warmup_rounds))

    @staticmethod
    def _grad_max_norm(param):
        if param.grad is None:
            return 0.0, 0.0
        grad = param.grad.detach().float()
        return grad.abs().max().item(), grad.norm().item()

    def run_spf_debug_checks(self, image, label, spf_scale, shared_lambda):
        prompt_learner = self.model.prompt_learner

        self.optim.zero_grad(set_to_none=True)
        output_local, output_fused, _, aux = self.model(image, spf_scale=spf_scale)
        loss_local = F.cross_entropy(output_local, label)
        loss_fused = F.cross_entropy(output_fused, label)
        loss_shared = shared_lambda * aux["shared_pull_loss"]
        (loss_local + loss_fused + loss_shared).backward()
        global_max, global_norm = self._grad_max_norm(prompt_learner.ctx_global)
        local_max, local_norm = self._grad_max_norm(prompt_learner.ctx_local)
        print(
            "[SPF Debug] personal+shared backward: "
            f"ctx_global max={global_max:.3e} norm={global_norm:.3e}, "
            f"ctx_local max={local_max:.3e} norm={local_norm:.3e}"
        )
        assert global_max < 1e-10, (
            "SPF gradient leakage: personal/shared loss updated ctx_global"
        )

        self.optim.zero_grad(set_to_none=True)
        _, _, output_global, _ = self.model(image, spf_scale=spf_scale)
        loss_global = F.cross_entropy(output_global, label)
        loss_global.backward()
        global_max, global_norm = self._grad_max_norm(prompt_learner.ctx_global)
        local_max, local_norm = self._grad_max_norm(prompt_learner.ctx_local)
        print(
            "[SPF Debug] global backward: "
            f"ctx_global max={global_max:.3e} norm={global_norm:.3e}, "
            f"ctx_local max={local_max:.3e} norm={local_norm:.3e}"
        )
        assert local_max < 1e-10, (
            "SPF gradient leakage: global loss updated ctx_local"
        )

        self.optim.zero_grad(set_to_none=True)
        self._spf_debug_checked = True


    def forward_backward(self, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.GL_SVDMSE.PREC

        if self.model.prompt_learner.use_spf:
            spf_scale = self.get_spf_warmup_scale(
                getattr(self, "global_epoch", -1)
            )
            shared_lambda = (
                self.cfg.TRAINER.GL_SVDMSE.SPF_SHARED_LAMBDA * spf_scale
            )

            if (
                self.cfg.TRAINER.GL_SVDMSE.SPF_DEBUG_CHECKS
                and not getattr(self, "_spf_debug_checked", False)
            ):
                self.run_spf_debug_checks(image, label, spf_scale, shared_lambda)

            if prec == "amp":
                with autocast():
                    output_local, output_fused, output_global, aux = self.model(
                        image,
                        spf_scale=spf_scale
                    )
                    loss_local = F.cross_entropy(output_local, label)
                    loss_fused = F.cross_entropy(output_fused, label)
                    loss_global = F.cross_entropy(output_global, label)
                    loss = (
                        loss_local
                        + loss_global
                        + loss_fused
                        + shared_lambda * aux["shared_pull_loss"]
                    )
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output_local, output_fused, output_global, aux = self.model(
                    image,
                    spf_scale=spf_scale
                )
                loss_local = F.cross_entropy(output_local, label)
                loss_fused = F.cross_entropy(output_fused, label)
                loss_global = F.cross_entropy(output_global, label)
                loss = (
                    loss_local
                    + loss_global
                    + loss_fused
                    + shared_lambda * aux["shared_pull_loss"]
                )
                self.model_backward_and_update(loss)

            loss_summary = {
                "loss": loss.item(),
                "loss_local": loss_local.item(),
                "loss_fused": loss_fused.item(),
                "loss_global": loss_global.item(),
                "loss_shared_scaled": float(shared_lambda * aux["shared_pull_loss"].detach().item()),
                "local_acc": compute_accuracy(output_local, label)[0].item(),
                "acc": compute_accuracy(output_fused, label)[0].item(),
                "fused_acc": compute_accuracy(output_fused, label)[0].item(),
                "global_acc": compute_accuracy(output_global, label)[0].item(),
                "spf_gamma": float(aux["gamma"].item()),
                "spf_gamma_base": float(aux["gamma_base"].item()),
                "spf_scale": float(aux["spf_scale"].item()),
                "spf_shared_lambda": float(shared_lambda),
                "spf_rank": float(aux["svd_rank"].item()),
                "spf_shared_loss": float(aux["shared_pull_loss"].item()),
                "spf_prompt_shared_mse": float(aux["prompt_shared_mse"].item()),
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

        return loss_summary

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
