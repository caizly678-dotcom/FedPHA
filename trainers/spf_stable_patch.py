"""Stable SPF training patch.

This module is imported by federated_spf_stable.py before build_trainer().
It keeps the original GL_SVDMSE implementation untouched and installs a
controlled SPF path with: (1) frozen per-client fusion anchors, (2) separate
fused/local/global supervision, and (3) optional low-weight shared alignment.
"""

import torch
from torch.nn import functional as F

from Dassl.dassl.metrics import compute_accuracy
from trainers.GL_SVDMSE import CustomCLIP, GL_SVDMSE, PromptLearner
from trainers.spf_utils import compute_shared_basis, project_to_basis


_ORIGINAL_PROMPT_INIT = PromptLearner.__init__
_ORIGINAL_PROMPT_FORWARD = PromptLearner.forward
_ORIGINAL_CLIP_FORWARD = CustomCLIP.forward
_ORIGINAL_TRAINER_FORWARD_BACKWARD = GL_SVDMSE.forward_backward


def _patched_prompt_init(self, *args, **kwargs):
    _ORIGINAL_PROMPT_INIT(self, *args, **kwargs)
    if self.use_spf and not hasattr(self, "spf_anchor_ctx"):
        # Persist the anchor so every client's fused model can be evaluated
        # against the exact global state used while training that local prompt.
        self.register_buffer("spf_anchor_ctx", self.ctx_global.detach().clone())


def _build_prompts_from_ctx(self, ctx):
    if self.class_token_position != "end":
        raise NotImplementedError("Stable SPF currently supports CLASS_TOKEN_POSITION == 'end'")
    ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
    ctx = ctx.permute(1, 0, 2, 3).contiguous().view(
        self.N * self.n_cls, self.n_ctx, ctx.shape[-1]
    )
    prefix = self.token_prefix.to(device=ctx.device, dtype=ctx.dtype)
    suffix = self.token_suffix.to(device=ctx.device, dtype=ctx.dtype)
    return torch.cat([prefix, ctx, suffix], dim=1)


@torch.no_grad()
def refresh_spf_anchor(self):
    if not self.use_spf:
        return
    self.spf_anchor_ctx.copy_(
        self.ctx_global.detach().to(
            device=self.spf_anchor_ctx.device,
            dtype=self.spf_anchor_ctx.dtype,
        )
    )


def fuse_ctx_spf_stable(self, ctx_local):
    # The anchor is a per-client snapshot of the server prompt received at the
    # beginning of local training. It is intentionally detached from fused CE.
    anchor_ctx = self.spf_anchor_ctx.to(device=ctx_local.device, dtype=ctx_local.dtype).detach()
    basis, _ = compute_shared_basis(
        anchor_ctx,
        energy=self.spf_energy,
        min_rank=self.spf_min_rank,
        max_rank=self.spf_max_rank,
    )
    local_shared = project_to_basis(ctx_local, basis)
    anchor_shared = project_to_basis(anchor_ctx, basis)

    gamma = self.fusion_gamma.to(device=ctx_local.device, dtype=ctx_local.dtype).view(1, 1, 1)
    correction = gamma * (anchor_shared - local_shared)
    fused_ctx = ctx_local + correction

    # This term is optional. Its default is zero in the stable runner because
    # forcing all local shared coordinates to match the anchor can erase client
    # specificity under label/domain skew.
    shared_pull_loss = F.mse_loss(local_shared.float(), anchor_shared.float())
    correction_ratio = correction.detach().float().norm() / ctx_local.detach().float().norm().clamp_min(1e-12)
    aux = {
        "shared_pull_loss": shared_pull_loss,
        "gamma": gamma.detach().float().mean(),
        "svd_rank": torch.tensor(float(basis.shape[1]), device=ctx_local.device),
        "correction_ratio": correction_ratio,
    }
    return fused_ctx, aux


def _patched_prompt_forward(self, forward_mode="fused"):
    if not self.use_spf:
        return _ORIGINAL_PROMPT_FORWARD(self)

    mode = {"personal": "fused", "local": "local_only", "global": "global_only"}.get(
        forward_mode, forward_mode
    )
    if mode == "fused":
        fused_ctx, aux = self.fuse_ctx_spf_stable(self.ctx_local)
        return self._build_prompts_from_ctx(fused_ctx), aux
    if mode == "local_only":
        return self._build_prompts_from_ctx(self.ctx_local)
    if mode == "global_only":
        return self._build_prompts_from_ctx(self.ctx_global)
    raise ValueError(f"Unsupported stable SPF forward mode: {forward_mode}")


def _logits_from_prompts(model, image_features, prompts):
    text_features = model.text_encoder(prompts, model.tokenized_prompts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model.logit_scale.exp() * image_features @ text_features.t()


def _patched_clip_forward(self, image, idx=None, forward_mode="fused"):
    if not self.prompt_learner.use_spf:
        return _ORIGINAL_CLIP_FORWARD(self, image, idx)

    image_features = self.image_encoder(image.type(self.dtype))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    if self.training and forward_mode == "train":
        fused_prompts, aux = self.prompt_learner(forward_mode="fused")
        local_prompts = self.prompt_learner(forward_mode="local_only")
        global_prompts = self.prompt_learner(forward_mode="global_only")
        logits_fused = _logits_from_prompts(self, image_features, fused_prompts)
        logits_local = _logits_from_prompts(self, image_features, local_prompts)
        logits_global = _logits_from_prompts(self, image_features, global_prompts)
        return logits_fused, logits_local, logits_global, aux

    prompt_out = self.prompt_learner(forward_mode=forward_mode)
    prompts = prompt_out[0] if isinstance(prompt_out, tuple) else prompt_out
    return _logits_from_prompts(self, image_features, prompts)


def _patched_forward_backward(self, batch_idx, batch, **kwargs):
    if not self.model.prompt_learner.use_spf:
        return _ORIGINAL_TRAINER_FORWARD_BACKWARD(self, batch_idx, batch, **kwargs)

    image, label = self.parse_batch_train(batch)
    cfg = self.cfg.TRAINER.GL_SVDMSE
    local_lambda = float(getattr(cfg, "SPF_LOCAL_LAMBDA", 0.5))
    global_lambda = float(getattr(cfg, "SPF_GLOBAL_LAMBDA", 1.0))
    shared_lambda = float(getattr(cfg, "SPF_SHARED_LAMBDA", 0.0))
    prec = cfg.PREC

    if prec == "amp":
        with torch.cuda.amp.autocast():
            output_fused, output_local, output_global, aux = self.model(image, forward_mode="train")
            loss_fused = F.cross_entropy(output_fused, label)
            loss_local = F.cross_entropy(output_local, label)
            loss_global = F.cross_entropy(output_global, label)
            loss = (
                loss_fused
                + local_lambda * loss_local
                + global_lambda * loss_global
                + shared_lambda * aux["shared_pull_loss"]
            )
        self.optim.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim)
        self.scaler.update()
    else:
        output_fused, output_local, output_global, aux = self.model(image, forward_mode="train")
        loss_fused = F.cross_entropy(output_fused, label)
        loss_local = F.cross_entropy(output_local, label)
        loss_global = F.cross_entropy(output_global, label)
        loss = (
            loss_fused
            + local_lambda * loss_local
            + global_lambda * loss_global
            + shared_lambda * aux["shared_pull_loss"]
        )
        self.model_backward_and_update(loss)

    if (self.batch_idx + 1) == self.num_batches:
        self.update_lr()

    return {
        "loss": float(loss.item()),
        "acc": float(compute_accuracy(output_fused, label)[0].item()),
        "local_acc": float(compute_accuracy(output_local, label)[0].item()),
        "global_acc": float(compute_accuracy(output_global, label)[0].item()),
        "loss_fused": float(loss_fused.item()),
        "loss_local": float(loss_local.item()),
        "loss_global": float(loss_global.item()),
        "spf_gamma": float(aux["gamma"].item()),
        "spf_rank": float(aux["svd_rank"].item()),
        "spf_shared_loss": float(aux["shared_pull_loss"].item()),
        "spf_correction_ratio": float(aux["correction_ratio"].item()),
    }


PromptLearner.__init__ = _patched_prompt_init
PromptLearner._build_prompts_from_ctx = _build_prompts_from_ctx
PromptLearner.refresh_spf_anchor = refresh_spf_anchor
PromptLearner.fuse_ctx_spf_stable = fuse_ctx_spf_stable
PromptLearner.forward = _patched_prompt_forward
CustomCLIP.forward = _patched_clip_forward
GL_SVDMSE.forward_backward = _patched_forward_backward
