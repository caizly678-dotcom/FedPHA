import os
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
        self.debug_input_dtype = False
        self.last_input_dtype = None

    def forward(self, prompts, tokenized_prompts):
        if self.debug_forward_count:
            self.forward_count += 1
        if self.debug_input_dtype:
            self.last_input_dtype = prompts.dtype

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
        self.spf_gate_type = cfg.TRAINER.GL_SVDMSE.SPF_GATE_TYPE
        self.spf_gamma_base = cfg.TRAINER.GL_SVDMSE.SPF_GAMMA_BASE
        self.spf_gamma_delta = cfg.TRAINER.GL_SVDMSE.SPF_GAMMA_DELTA
        self.spf_gamma_min = cfg.TRAINER.GL_SVDMSE.SPF_GAMMA_MIN
        self.spf_gamma_neg_min = cfg.TRAINER.GL_SVDMSE.SPF_GAMMA_NEG_MIN
        self.spf_bipolar_bias_init = cfg.TRAINER.GL_SVDMSE.SPF_BIPOLAR_BIAS_INIT
        self.spf_rank_gate_lr_mult = cfg.TRAINER.GL_SVDMSE.SPF_GATE_LR_MULT
        self.spf_bipolar_gate = cfg.TRAINER.GL_SVDMSE.SPF_BIPOLAR_GATE
        self.use_dynamic_gate = cfg.TRAINER.GL_SVDMSE.SPF_USE_DYNAMIC_GATE
        self.alignment_type = cfg.TRAINER.GL_SVDMSE.SPF_ALIGNMENT_TYPE
        self.freeze_anchor = cfg.TRAINER.GL_SVDMSE.SPF_FREEZE_ANCHOR
        self.spf_push_alpha = cfg.TRAINER.GL_SVDMSE.SPF_PUSH_ALPHA
        self.shared_init = cfg.TRAINER.GL_SVDMSE.SPF_SHARED_INIT
        self.fixed_round_basis = cfg.TRAINER.GL_SVDMSE.SPF_FIXED_ROUND_BASIS
        self.debug_gate = bool(
            self.get_spf_method_label().startswith("A2")
            and (
                getattr(cfg.TRAINER.GL_SVDMSE, "SPF_DEBUG_GATE", False)
                or os.environ.get("SPF_DEBUG_GATE", "").lower() in {"1", "true", "yes", "on"}
            )
        )
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
            if self.spf_gate_type != "rankwise":
                raise ValueError(f"Unsupported SPF_GATE_TYPE: {self.spf_gate_type}")
            if self.spf_bipolar_gate:
                if not (self.spf_gamma_neg_min < 0.0 < self.spf_gamma_max):
                    raise ValueError("Bipolar SPF gamma range must satisfy neg_min < 0 < max")
            elif not (0.0 <= self.spf_gamma_min <= self.spf_gamma_base <= self.spf_gamma_max):
                raise ValueError("SPF gamma range must satisfy min <= base <= max")
            if self.spf_gamma_delta <= 0:
                raise ValueError("SPF_GAMMA_DELTA must be positive")
            self.rank_gate = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, 1),
            )
            nn.init.normal_(self.rank_gate[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.rank_gate[-1].bias)
            self.rank_bias = nn.Parameter(torch.zeros(self.spf_max_rank, dtype=torch.float32))
            if self.spf_bipolar_gate:
                nn.init.constant_(self.rank_bias, self.spf_bipolar_bias_init)
            print(f"SPF gate type: {self.spf_gate_type}")
            print("SPF rank_gate parameters:")
            for name, param in self.rank_gate.named_parameters():
                print(f"  rank_gate.{name}: numel={param.numel()}, requires_grad={param.requires_grad}")
            print(f"  rank_bias: numel={self.rank_bias.numel()}, requires_grad={self.rank_bias.requires_grad}")
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
        if not self.use_dynamic_gate and not self.freeze_anchor and self.alignment_type == "cosine":
            return "A0"
        if not self.use_dynamic_gate and self.freeze_anchor and self.alignment_type == "cosine":
            return "A1"
        if self.use_dynamic_gate and self.freeze_anchor and self.alignment_type == "cosine":
            return f"A2-{self.spf_gate_type}"
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
        basis_float = basis.float()
        ctx_local_float = ctx_local.float()
        local_coord = ctx_local_float @ basis_float
        global_coord = anchor_coord.to(device=local_coord.device, dtype=torch.float32).detach()
        delta_coord = global_coord - local_coord
        local_shared = local_coord @ basis_float.t()
        global_shared = global_coord @ basis_float.t()
        residual = global_shared - local_shared

        if self.use_dynamic_gate:
            r = basis.shape[1]
            anchor_coord32 = anchor_coord.to(device=local_coord.device, dtype=torch.float32)
            eps = 1e-12
            local_rank = local_coord.float().transpose(0, -1).reshape(r, -1)
            anchor_rank = anchor_coord32.float().transpose(0, -1).reshape(r, -1)
            delta_rank = delta_coord.float().transpose(0, -1).reshape(r, -1)
            compatibility = F.cosine_similarity(local_rank, anchor_rank, dim=-1)
            relative_gap = delta_rank.norm(dim=-1) / local_rank.norm(dim=-1).clamp_min(eps)
            gate_input = torch.stack(
                [compatibility, torch.log1p(relative_gap)],
                dim=-1,
            ).detach()
            raw = self.rank_gate(gate_input).squeeze(-1) + self.rank_bias[:r]
            if self.spf_bipolar_gate:
                bias_origin = math.tanh(self.spf_bipolar_bias_init)
                gamma = self.spf_gamma_base + self.spf_gamma_delta * (
                    torch.tanh(raw) - bias_origin
                )
                gamma = gamma.clamp(self.spf_gamma_neg_min, self.spf_gamma_max)
            else:
                gamma = self.spf_gamma_base + self.spf_gamma_delta * torch.tanh(raw)
                gamma = gamma.clamp(self.spf_gamma_min, self.spf_gamma_max)
        else:
            r = basis.shape[1]
            gamma = torch.full((r,), self.spf_gamma_init, device=ctx_local.device, dtype=torch.float32)

        correction_coord = delta_coord * gamma.view(1, 1, r)
        correction32 = correction_coord @ basis_float.t()
        fused_dynamic_fp32 = ctx_local_float + correction32
        fixed_gamma = torch.full_like(gamma.detach(), self.spf_gamma_base)
        fixed_correction32 = (delta_coord * fixed_gamma.view(1, 1, r)) @ basis_float.t()
        fused_fixed_gamma_fp32 = ctx_local_float + fixed_correction32
        fused_ctx = fused_dynamic_fp32.to(dtype=ctx_local.dtype)
        fused_fixed_cast = fused_fixed_gamma_fp32.to(dtype=ctx_local.dtype)
        effective_dynamic_delta_fp32 = (fused_dynamic_fp32 - fused_fixed_gamma_fp32).detach().float().norm()
        effective_dynamic_delta_after_cast = (fused_ctx.float() - fused_fixed_cast.float()).detach().norm()

        if self.debug_gate and self.use_dynamic_gate and self.training and torch.is_grad_enabled():
            assert gamma.requires_grad is True
            assert fused_dynamic_fp32.requires_grad is True
            if residual.detach().float().norm().item() > 0.0:
                assert raw.requires_grad is True
                assert raw.grad_fn is not None
                assert gamma.grad_fn is not None

        if self.alignment_type != "cosine":
            raise ValueError("SPF pull loss only supports cosine alignment; MSE is deprecated")
        anchor_coord = anchor_coord.to(device=local_coord.device, dtype=local_coord.dtype).detach()
        shared_pull_loss = 1.0 - F.cosine_similarity(
            local_coord.float().flatten().unsqueeze(0),
            anchor_coord.float().flatten().unsqueeze(0),
            dim=-1,
        ).mean()
        push_distance = torch.norm(
            ctx_local_float - ctx_global.float().detach(),
            dim=-1,
        )
        push_loss = F.relu(self.spf_push_alpha - push_distance).mean()
        correction_ratio = correction32.detach().float().norm() / ctx_local.detach().float().norm().clamp_min(1e-12)
        negative_gamma_ratio = (gamma.detach().float() < 0).float().mean()

        aux = {
            "shared_pull_loss": shared_pull_loss,
            "push_loss": push_loss,
            "gamma_mean": gamma.detach().float().mean(),
            "gamma_std": gamma.detach().float().std(unbiased=False),
            "gamma_min": gamma.detach().float().min(),
            "gamma_max": gamma.detach().float().max(),
            "negative_gamma_ratio": negative_gamma_ratio,
            "correction_ratio": correction_ratio,
            "svd_rank": torch.tensor(float(basis.shape[1]), device=ctx_local.device),
            "rank_gamma": gamma.detach().float(),
            "effective_dynamic_delta_fp32": effective_dynamic_delta_fp32,
            "effective_dynamic_delta_after_cast": effective_dynamic_delta_after_cast,
            "fused_ctx_dtype_is_fp32": torch.tensor(
                float(fused_dynamic_fp32.dtype == torch.float32), device=ctx_local.device
            ),
        }
        if self.debug_gate and self.use_dynamic_gate and self.training and torch.is_grad_enabled():
            aux.update(
                {
                    "gamma": gamma,
                    "fused_ctx": fused_dynamic_fp32,
                    "residual": residual,
                    "gate_input": gate_input.detach().float(),
                    "compatibility": compatibility.detach().float(),
                    "relative_gap": relative_gap.detach().float(),
                    "gate_input_token_std": gate_input.detach().float().std(dim=0, unbiased=False).mean(),
                    "max_abs_gamma_delta": (gamma.detach().float() - self.spf_gamma_base).abs().max(),
                }
            )
        if self.debug_gate and self.use_dynamic_gate and effective_dynamic_delta_after_cast.item() == 0.0:
            print(
                "[SPF_DEBUG_GATE][WARNING] effective_dynamic_delta_after_cast is zero; "
                "this batch is not a valid dynamic Gate experiment"
            )
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

    def _build_prompts_from_ctx(self, ctx):
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        ctx = ctx.permute(1, 0, 2, 3).contiguous().view(
            self.N * self.n_cls, self.n_ctx, ctx.shape[-1]
        )
        prefix = self.token_prefix.to(device=ctx.device, dtype=ctx.dtype)
        suffix = self.token_suffix.to(device=ctx.device, dtype=ctx.dtype)
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self, forward_mode="personal"):
        if forward_mode == "global_only":
            if self.class_token_position != "end":
                raise NotImplementedError("global_only only supports CLASS_TOKEN_POSITION == 'end'")
            return self._build_prompts_from_ctx(self.ctx_global)
        if forward_mode != "personal":
            raise ValueError(f"Unsupported prompt forward mode: {forward_mode}")

        if self.use_spf:
            if self.class_token_position != "end":
                raise NotImplementedError("SPF-FedPHA only supports CLASS_TOKEN_POSITION == 'end'")

            fused_ctx, aux = self.fuse_ctx_spf(self.ctx_local, self.ctx_global)

            prompts = self._build_prompts_from_ctx(fused_ctx)
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
        self.text_encoder.debug_input_dtype = (
            cfg.TRAINER.GL_SVDMSE.USE_SPF
            and cfg.TRAINER.GL_SVDMSE.SPF_USE_DYNAMIC_GATE
            and cfg.TRAINER.GL_SVDMSE.SPF_DEBUG_GATE
        )
        self.N = cfg.TRAINER.GL_SVDMSE.N

    def _encode_image(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def _logits_from_prompts(self, image_features, prompts, tokenized_prompts):
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        return logit_scale * image_features @ text_features.t()

    def _forward_prompts(self, image, prompts, tokenized_prompts):
        image_features = self._encode_image(image)
        return self._logits_from_prompts(image_features, prompts, tokenized_prompts)

    def forward(self, image, idx=None, forward_mode="personal"):
        tokenized_prompts = self.tokenized_prompts
        out = self.prompt_learner(forward_mode=forward_mode)

        if forward_mode == "global_only":
            return self._forward_prompts(image, out, tokenized_prompts)

        if self.prompt_learner.use_spf:
            prompts, aux = out
            image_features = self._encode_image(image)
            logits = self._logits_from_prompts(image_features, prompts, tokenized_prompts)
            if self.training:
                global_prompts = self.prompt_learner(forward_mode="global_only")
                logits_global = self._logits_from_prompts(
                    image_features,
                    global_prompts,
                    tokenized_prompts,
                )
            text_encoder = self.text_encoder.module if hasattr(self.text_encoder, "module") else self.text_encoder
            aux["text_encoder_input_dtype_is_fp32"] = torch.tensor(
                float(text_encoder.last_input_dtype == torch.float32),
                device=logits.device,
            )

            if self.training:
                return logits, logits_global, aux

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

    def _build_prompt_optimizer(self):
        prompt_learner = self.model.prompt_learner
        cfg = self.cfg
        if (
            prompt_learner.use_spf
            and prompt_learner.use_dynamic_gate
            and hasattr(prompt_learner, "rank_gate")
        ):
            base_params = [prompt_learner.ctx_global, prompt_learner.ctx_local]
            gate_params = list(prompt_learner.rank_gate.parameters()) + [prompt_learner.rank_bias]
            param_groups = [
                {"params": base_params, "lr": cfg.OPTIM.LR},
                {"params": gate_params, "lr": cfg.OPTIM.LR * cfg.TRAINER.GL_SVDMSE.SPF_GATE_LR_MULT},
            ]
            return build_optimizer(prompt_learner, cfg.OPTIM, param_groups=param_groups)
        return build_optimizer(prompt_learner, cfg.OPTIM)

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
        self.optim = self._build_prompt_optimizer()
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.GL_SVDMSE.PREC == "amp" else None

    def _debug_gate_enabled(self):
        prompt_learner = self.model.prompt_learner
        return bool(
            prompt_learner.get_spf_method_label().startswith("A2")
            and (
                getattr(self.cfg.TRAINER.GL_SVDMSE, "SPF_DEBUG_GATE", False)
                or os.environ.get("SPF_DEBUG_GATE", "").lower() in {"1", "true", "yes", "on"}
            )
        )

    def _debug_gate_last_layer(self):
        prompt_learner = self.model.prompt_learner
        if not hasattr(prompt_learner, "rank_gate"):
            return None
        return prompt_learner.rank_gate[-1]

    def _debug_gate_param_snapshot(self):
        last = self._debug_gate_last_layer()
        if last is None:
            return None
        return {
            "last_weight": last.weight.detach().clone(),
            "last_bias": last.bias.detach().clone(),
            "rank_bias": self.model.prompt_learner.rank_bias.detach().clone()
            if hasattr(self.model.prompt_learner, "rank_bias")
            else None,
        }

    def _debug_gate_log_grads_before_step(self, client_id=None, global_epoch=None):
        if not self._debug_gate_enabled():
            return
        prompt_learner = self.model.prompt_learner
        if not hasattr(prompt_learner, "rank_gate"):
            return
        batch_no = int(getattr(self, "batch_idx", 0)) + 1
        prefix = f"[SPF_DEBUG_GATE][client={client_id}][round={global_epoch}][batch={batch_no}][before_step]"
        for name, param in prompt_learner.rank_gate.named_parameters():
            grad_is_none = param.grad is None
            grad_norm = float("nan") if grad_is_none else float(param.grad.detach().float().norm().cpu().item())
            print(
                f"{prefix} rank_gate.{name} grad_is_none={grad_is_none} "
                f"grad_norm={grad_norm:.8e}"
            )
        grad_is_none = prompt_learner.rank_bias.grad is None
        grad_norm = float("nan") if grad_is_none else float(prompt_learner.rank_bias.grad.detach().float().norm().cpu().item())
        print(f"{prefix} rank_bias grad_is_none={grad_is_none} grad_norm={grad_norm:.8e}")

    def _debug_gate_log_step_change(self, before, client_id=None, global_epoch=None):
        if not self._debug_gate_enabled() or before is None:
            return
        last = self._debug_gate_last_layer()
        weight_after = last.weight.detach()
        bias_after = last.bias.detach()
        weight_delta = (weight_after - before["last_weight"]).float().abs().max()
        bias_delta = (bias_after - before["last_bias"]).float().abs().max()
        rank_bias_delta = torch.tensor(0.0, device=weight_after.device)
        if before.get("rank_bias") is not None and hasattr(self.model.prompt_learner, "rank_bias"):
            rank_bias_delta = (
                self.model.prompt_learner.rank_bias.detach() - before["rank_bias"]
            ).float().abs().max()
        max_abs_parameter_change = max(
            float(weight_delta.cpu().item()),
            float(bias_delta.cpu().item()),
            float(rank_bias_delta.cpu().item()),
        )
        batch_no = int(getattr(self, "batch_idx", 0)) + 1
        print(
            f"[SPF_DEBUG_GATE][client={client_id}][round={global_epoch}][batch={batch_no}][after_step] "
            f"weight_norm_before={float(before['last_weight'].float().norm().cpu().item()):.8e} "
            f"weight_norm_after={float(weight_after.float().norm().cpu().item()):.8e} "
            f"bias_before={float(before['last_bias'].float().mean().cpu().item()):.8e} "
            f"bias_after={float(bias_after.float().mean().cpu().item()):.8e} "
            f"rank_bias_delta={float(rank_bias_delta.cpu().item()):.8e} "
            f"max_abs_parameter_change={max_abs_parameter_change:.8e}"
        )

    def forward_backward(self, batch_idx, batch, **kwargs):
        client_id = kwargs.get("idx", kwargs.get("client_id", getattr(self, "_spf_debug_client_id", None)))
        global_epoch = kwargs.get("global_epoch", getattr(self, "_spf_debug_global_epoch", None))
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.GL_SVDMSE.PREC
        self.model.image_encoder.eval()
        self.model.text_encoder.eval()
        if self.cfg.TRAINER.GL_SVDMSE.DEBUG_TEXT_ENCODER_FORWARD_COUNT:
            self.model.text_encoder.forward_count = 0

        if self.model.prompt_learner.use_spf:
            if prec == "amp":
                with autocast():
                    output, output_global, aux = self.model(image)
                if self.cfg.TRAINER.GL_SVDMSE.DEBUG_TEXT_ENCODER_FORWARD_COUNT:
                    assert self.model.text_encoder.forward_count == 2
                    print("text_encoder_forwards_per_batch=2")
                loss_ce = F.cross_entropy(output, label)
                loss_ce_global = F.cross_entropy(output_global, label)
                loss = (
                    loss_ce
                    + self.cfg.TRAINER.GL_SVDMSE.SPF_GLOBAL_LAMBDA * loss_ce_global
                    + self.cfg.TRAINER.GL_SVDMSE.SPF_SHARED_LAMBDA * aux["shared_pull_loss"]
                    + self.cfg.TRAINER.GL_SVDMSE.SPF_PUSH_LAMBDA * aux["push_loss"]
                )
                self.optim.zero_grad()
                before_step = self._debug_gate_param_snapshot()
                self.scaler.scale(loss).backward()
                if self._debug_gate_enabled():
                    self.scaler.unscale_(self.optim)
                    self._debug_gate_log_grads_before_step(client_id, global_epoch)
                self.scaler.step(self.optim)
                self.scaler.update()
                self._debug_gate_log_step_change(before_step, client_id, global_epoch)
            else:
                output, output_global, aux = self.model(image)
                if self.cfg.TRAINER.GL_SVDMSE.DEBUG_TEXT_ENCODER_FORWARD_COUNT:
                    assert self.model.text_encoder.forward_count == 2
                    print("text_encoder_forwards_per_batch=2")
                loss_ce = F.cross_entropy(output, label)
                loss_ce_global = F.cross_entropy(output_global, label)
                loss = (
                    loss_ce
                    + self.cfg.TRAINER.GL_SVDMSE.SPF_GLOBAL_LAMBDA * loss_ce_global
                    + self.cfg.TRAINER.GL_SVDMSE.SPF_SHARED_LAMBDA * aux["shared_pull_loss"]
                    + self.cfg.TRAINER.GL_SVDMSE.SPF_PUSH_LAMBDA * aux["push_loss"]
                )
                if self._debug_gate_enabled():
                    self.optim.zero_grad()
                    before_step = self._debug_gate_param_snapshot()
                    self.detect_anomaly(loss)
                    loss.backward()
                    self._debug_gate_log_grads_before_step(client_id, global_epoch)
                    self.optim.step()
                    self._debug_gate_log_step_change(before_step, client_id, global_epoch)
                else:
                    self.model_backward_and_update(loss)

            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
                "global_acc": compute_accuracy(output_global, label)[0].item(),
                "loss_ce_fused": float(loss_ce.item()),
                "loss_ce_global": float(loss_ce_global.item()),
                "spf_shared_loss": float(aux["shared_pull_loss"].item()),
                "spf_push_loss": float(aux["push_loss"].item()),
                "spf_gamma_mean": float(aux["gamma_mean"].item()),
                "spf_gamma_std": float(aux["gamma_std"].item()),
                "spf_gamma_min": float(aux["gamma_min"].item()),
                "spf_gamma_max": float(aux["gamma_max"].item()),
                "spf_negative_gamma_ratio": float(aux["negative_gamma_ratio"].item()),
                "spf_correction_ratio": float(aux["correction_ratio"].item()),
                "spf_rank": float(aux["svd_rank"].item()),
                "spf_fused_ctx_dtype_is_fp32": float(aux["fused_ctx_dtype_is_fp32"].item()),
                "spf_text_encoder_input_dtype_is_fp32": float(aux["text_encoder_input_dtype_is_fp32"].item()),
                "spf_effective_dynamic_delta_fp32": float(aux["effective_dynamic_delta_fp32"].item()),
                "spf_effective_dynamic_delta_after_cast": float(aux["effective_dynamic_delta_after_cast"].item()),
            }
            if not hasattr(self, "_last_spf_train_metrics"):
                self._last_spf_train_metrics = []
            self._last_spf_train_metrics.append(loss_summary.copy())
            if self._debug_gate_enabled():
                gate_input_token_std = aux.get("gate_input_token_std", torch.tensor(float("nan")))
                max_abs_gamma_delta = aux.get("max_abs_gamma_delta", torch.tensor(float("nan")))
                gate_input_token_std = float(gate_input_token_std.detach().cpu().item())
                max_abs_gamma_delta = float(max_abs_gamma_delta.detach().cpu().item())
                rank_gamma = aux["rank_gamma"].detach().float().cpu().tolist()
                compatibility = aux.get("compatibility", torch.empty(0)).detach().float().cpu().tolist()
                relative_gap = aux.get("relative_gap", torch.empty(0)).detach().float().cpu().tolist()
                rank_gamma_str = "[" + ", ".join(f"{value:.8f}" for value in rank_gamma) + "]"
                compatibility_str = "[" + ", ".join(f"{value:.8f}" for value in compatibility) + "]"
                relative_gap_str = "[" + ", ".join(f"{value:.8f}" for value in relative_gap) + "]"
                print(
                    f"[SPF_DEBUG_GATE][client={client_id}][round={global_epoch}]"
                    f"[batch={batch_idx + 1}] rank_gamma_mean={loss_summary['spf_gamma_mean']:.8f} "
                    f"rank_gamma_std={loss_summary['spf_gamma_std']:.8e} "
                    f"rank_gamma_min={loss_summary['spf_gamma_min']:.8f} "
                    f"rank_gamma_max={loss_summary['spf_gamma_max']:.8f} "
                    f"rank_gamma={rank_gamma_str} "
                    f"max_abs_gamma_delta={max_abs_gamma_delta:.8e} "
                    f"gate_input_token_std={gate_input_token_std:.8e} "
                    f"compatibility={compatibility_str} "
                    f"relative_gap={relative_gap_str} "
                    f"effective_dynamic_delta_fp32={loss_summary['spf_effective_dynamic_delta_fp32']:.8e} "
                    f"effective_dynamic_delta_after_cast={loss_summary['spf_effective_dynamic_delta_after_cast']:.8e} "
                    f"correction_ratio={loss_summary['spf_correction_ratio']:.8e} "
                    f"rank={loss_summary['spf_rank']:.0f} "
                    f"fused_ctx_dtype_is_fp32={loss_summary['spf_fused_ctx_dtype_is_fp32']:.0f} "
                    f"text_encoder_input_dtype_is_fp32={loss_summary['spf_text_encoder_input_dtype_is_fp32']:.0f}"
                )
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
