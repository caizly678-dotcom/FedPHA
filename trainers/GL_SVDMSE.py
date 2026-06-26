import os.path as osp
import math
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
        self.debug_forward_count = False
        self.forward_count = 0

    def forward(self, prompts, tokenized_prompts):
        if self.debug_forward_count:
            self.forward_count += 1

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
        self.spf_gamma_init = cfg.TRAINER.GL_SVDMSE.SPF_GAMMA_INIT
        self.spf_gamma_max = cfg.TRAINER.GL_SVDMSE.SPF_GAMMA_MAX
        self.spf_gate_hidden_ratio = cfg.TRAINER.GL_SVDMSE.SPF_GATE_HIDDEN_RATIO
        self.use_dynamic_gate = cfg.TRAINER.GL_SVDMSE.SPF_USE_DYNAMIC_GATE
        self.alignment_type = cfg.TRAINER.GL_SVDMSE.SPF_ALIGNMENT_TYPE
        self.freeze_anchor = cfg.TRAINER.GL_SVDMSE.SPF_FREEZE_ANCHOR
        self.shared_init = cfg.TRAINER.GL_SVDMSE.SPF_SHARED_INIT
        self.fixed_round_basis = cfg.TRAINER.GL_SVDMSE.SPF_FIXED_ROUND_BASIS
        self.register_buffer("round_shared_basis", None, persistent=False)
        self._spf_basis = None
        self._spf_global_anchor_coord = None
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
            if self.shared_init:
                ctx_local = ctx_global.clone()
            else:
                ctx_local = ctx_vectors.unsqueeze(0).repeat(self.N, 1, 1)

        else:
            # random initialization
            if cfg.TRAINER.GL_SVDMSE.CSC:
                print("Initializing class-specific contexts")
                ctx_global = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_global, std=0.02)
                if self.shared_init:
                    ctx_local = ctx_global.clone()
                else:
                    ctx_local = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                    nn.init.normal_(ctx_local, std=0.02)
            else:
                print("Initializing a generic context")
                ctx_global = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_global, std=0.02)   # define the prompt to be trained
                if self.shared_init:
                    ctx_local = ctx_global.clone()
                else:
                    ctx_local = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
                    nn.init.normal_(ctx_local, std=0.02)   # define the prompt to be trained
            
            prompt_prefix = " ".join(["X"] * n_ctx)    

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"SPF method label: {self.get_spf_method_label()}")

        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_global = nn.Parameter(ctx_global)
        self.ctx_local = nn.Parameter(ctx_local)
        if self.use_spf and self.use_dynamic_gate:
            if not (0.0 < self.spf_gamma_init < self.spf_gamma_max):
                raise ValueError("SPF_GAMMA_INIT must be in (0, SPF_GAMMA_MAX)")
            gate_hidden_dim = max(ctx_dim // int(self.spf_gate_hidden_ratio), 1)
            self.gamma_net = nn.Sequential(
                nn.Linear(2 * ctx_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            nn.init.zeros_(self.gamma_net[-1].weight)
            gamma_ratio = self.spf_gamma_init / self.spf_gamma_max
            nn.init.constant_(self.gamma_net[-1].bias, math.log(gamma_ratio / (1.0 - gamma_ratio)))
            print("SPF gamma_net parameters:")
            for name, param in self.gamma_net.named_parameters():
                print(f"  gamma_net.{name}: numel={param.numel()}, requires_grad={param.requires_grad}")
        if self.shared_init:
            assert torch.allclose(self.ctx_global, self.ctx_local)
            assert self.ctx_global.data_ptr() != self.ctx_local.data_ptr()
        if "round_shared_basis" in self.state_dict():
            raise AssertionError("round_shared_basis must stay out of state_dict")
        
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

    def get_spf_method_label(self):
        if not self.use_spf:
            return "non-SPF"
        if not self.use_dynamic_gate and not self.freeze_anchor and self.alignment_type == "mse":
            return "A0"
        if not self.use_dynamic_gate and self.freeze_anchor and self.alignment_type == "mse":
            return "A1"
        if self.use_dynamic_gate and self.freeze_anchor and self.alignment_type == "mse":
            return "A2"
        if not self.use_dynamic_gate and self.freeze_anchor and self.alignment_type == "cosine":
            return "A3"
        if self.use_dynamic_gate and self.freeze_anchor and self.alignment_type == "cosine":
            return "A4"
        return (
            f"custom(dynamic_gate={self.use_dynamic_gate}, "
            f"freeze_anchor={self.freeze_anchor}, alignment={self.alignment_type})"
        )

    def _compute_basis_and_anchor(self, ctx_global):
        basis, _ = compute_shared_basis(
            ctx_global,
            energy=self.spf_energy,
            min_rank=self.spf_min_rank,
            max_rank=self.spf_max_rank,
        )
        basis = basis.detach()
        anchor_coord = (ctx_global.detach() @ basis).detach()
        return basis, anchor_coord

    @torch.no_grad()
    def refresh_round_shared_basis(self):
        basis, _ = compute_shared_basis(
            self.ctx_global,
            energy=self.spf_energy,
            min_rank=self.spf_min_rank,
            max_rank=self.spf_max_rank,
        )
        self.round_shared_basis = basis.detach().clone()
        return self.round_shared_basis

    @torch.no_grad()
    def refresh_spf_anchor(self):
        basis, anchor_coord = self._compute_basis_and_anchor(self.ctx_global)
        self._spf_basis = basis
        assert self._spf_basis.requires_grad is False
        self._spf_global_anchor_coord = anchor_coord
        return self._spf_basis

    def get_spf_basis(self, ctx_global=None):
        if not self.freeze_anchor:
            if ctx_global is None:
                ctx_global = self.ctx_global
            basis, anchor_coord = self._compute_basis_and_anchor(ctx_global)
            return basis, anchor_coord
        if self._spf_basis is None or self._spf_global_anchor_coord is None:
            raise RuntimeError("SPF anchor is not initialized; call refresh_spf_anchor() before local training")
        assert self._spf_basis.requires_grad is False
        return self._spf_basis, self._spf_global_anchor_coord

    def fuse_ctx_spf(self, ctx_local, ctx_global):
        basis, anchor_coord = self.get_spf_basis(ctx_global)
        local_coord = ctx_local @ basis
        global_coord = ctx_global @ basis
        local_shared = local_coord @ basis.t()
        global_shared = global_coord @ basis.t()
        residual = global_shared - local_shared

        if self.use_dynamic_gate:
            gate_input = torch.cat([ctx_local.detach(), residual.detach()], dim=-1)
            gate_dtype = next(self.gamma_net.parameters()).dtype
            raw_gamma = self.gamma_net(gate_input.to(dtype=gate_dtype))
            gamma = self.spf_gamma_max * torch.sigmoid(raw_gamma)
            gamma = gamma.to(device=ctx_local.device, dtype=ctx_local.dtype)
        else:
            gamma = torch.full_like(residual[..., :1], self.spf_gamma_init)
        correction = gamma * residual
        fused_ctx = ctx_local + correction

        anchor_coord = anchor_coord.to(device=local_coord.device, dtype=local_coord.dtype)
        if self.alignment_type == "cosine":
            shared_pull_loss = 1.0 - F.cosine_similarity(
                local_coord.float().flatten().unsqueeze(0),
                anchor_coord.float().flatten().unsqueeze(0),
                dim=-1,
            ).mean()
        elif self.alignment_type == "mse":
            shared_pull_loss = F.mse_loss(local_coord, anchor_coord.detach())
        else:
            raise ValueError(f"Unsupported SPF alignment type: {self.alignment_type}")
        correction_ratio = correction.detach().float().norm() / ctx_local.detach().float().norm().clamp_min(1e-12)

        aux = {
            "shared_pull_loss": shared_pull_loss,
            "gamma_mean": gamma.detach().float().mean(),
            "gamma_std": gamma.detach().float().std(unbiased=False),
            "gamma_min": gamma.detach().float().min(),
            "gamma_max": gamma.detach().float().max(),
            "correction_ratio": correction_ratio,
            "svd_rank": torch.tensor(float(basis.shape[1]), device=ctx_local.device),
        }
        return fused_ctx, aux

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

            fused_ctx, aux = self.fuse_ctx_spf(self.ctx_local, self.ctx_global)

            fused_ctx = fused_ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            fused_ctx = fused_ctx.permute(1, 0, 2, 3).contiguous().view(
                self.N * self.n_cls, self.n_ctx, fused_ctx.shape[-1]
            )
            prompts = torch.cat([self.token_prefix, fused_ctx, self.token_suffix], dim=1)
            return prompts, aux

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
        self.text_encoder.debug_forward_count = cfg.TRAINER.GL_SVDMSE.DEBUG_TEXT_ENCODER_FORWARD_COUNT
        self.N = cfg.TRAINER.GL_SVDMSE.N

    def forward(self, image, idx=None):
        tokenized_prompts = self.tokenized_prompts
        out = self.prompt_learner()

        if self.prompt_learner.use_spf:
            prompts, aux = out
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            if self.training:
                return logits, aux

            return logits

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
        print(f"SPF method label: {self.model.prompt_learner.get_spf_method_label()}")

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
        self.model.image_encoder.eval()
        self.model.text_encoder.eval()
        if self.cfg.TRAINER.GL_SVDMSE.DEBUG_TEXT_ENCODER_FORWARD_COUNT:
            self.model.text_encoder.forward_count = 0

        if self.model.prompt_learner.use_spf:
            if prec == "amp":
                with autocast():
                    output, aux = self.model(image)
                if self.cfg.TRAINER.GL_SVDMSE.DEBUG_TEXT_ENCODER_FORWARD_COUNT:
                    assert self.model.text_encoder.forward_count == 1
                    print("text_encoder_forwards_per_batch=1")
                loss_ce = F.cross_entropy(output, label)
                loss = (
                    loss_ce
                    + self.cfg.TRAINER.GL_SVDMSE.SPF_SHARED_LAMBDA * aux["shared_pull_loss"]
                )
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output, aux = self.model(image)
                if self.cfg.TRAINER.GL_SVDMSE.DEBUG_TEXT_ENCODER_FORWARD_COUNT:
                    assert self.model.text_encoder.forward_count == 1
                    print("text_encoder_forwards_per_batch=1")
                loss_ce = F.cross_entropy(output, label)
                loss = (
                    loss_ce
                    + self.cfg.TRAINER.GL_SVDMSE.SPF_SHARED_LAMBDA * aux["shared_pull_loss"]
                )
                self.model_backward_and_update(loss)

            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
                "spf_shared_loss": float(aux["shared_pull_loss"].item()),
                "spf_gamma_mean": float(aux["gamma_mean"].item()),
                "spf_gamma_std": float(aux["gamma_std"].item()),
                "spf_gamma_min": float(aux["gamma_min"].item()),
                "spf_gamma_max": float(aux["gamma_max"].item()),
                "spf_correction_ratio": float(aux["correction_ratio"].item()),
                "spf_rank": float(aux["svd_rank"].item()),
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
