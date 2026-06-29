import copy
import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from federated_main import (  # noqa: E402
    clone_spf_private_state,
    extract_spf_global_state,
    load_spf_client_state,
    rebuild_prompt_optimizer,
)
from trainers.GL_SVDMSE import CustomCLIP, PromptLearner  # noqa: E402


def make_cfg(
    use_spf=True,
    gamma_init=0.05,
    gamma_max=0.10,
    n_ctx=4,
    n_prompt=1,
    ctx_init=False,
    shared_init=False,
    class_token_position="end",
):
    trainer_cfg = SimpleNamespace(
        N_CTX=n_ctx,
        CSC=False,
        CTX_INIT=ctx_init,
        PREC="fp32",
        CLASS_TOKEN_POSITION=class_token_position,
        N=n_prompt,
        lambda_orthogonal=1.0,
        alpha=1.0,
        ratio=0.8,
        USE_SPF=use_spf,
        SPF_ENERGY=0.90,
        SPF_MIN_RANK=1,
        SPF_MAX_RANK=8,
        SPF_GAMMA_INIT=gamma_init,
        SPF_GAMMA_MAX=gamma_max,
        SPF_GATE_TYPE="rankwise",
        SPF_GAMMA_BASE=0.05,
        SPF_GAMMA_DELTA=0.03,
        SPF_GAMMA_MIN=0.005,
        SPF_GAMMA_NEG_MIN=-0.05,
        SPF_BIPOLAR_BIAS_INIT=0.2554,
        SPF_GATE_LR_MULT=3.0,
        SPF_BIPOLAR_GATE=False,
        SPF_USE_DYNAMIC_GATE=True,
        SPF_ALIGNMENT_TYPE="cosine",
        SPF_FREEZE_ANCHOR=True,
        SPF_SHARED_LAMBDA=0.1,
        SPF_GLOBAL_LAMBDA=1.0,
        SPF_PUSH_LAMBDA=0.0,
        SPF_PUSH_ALPHA=1.0,
        SPF_SHARED_INIT=shared_init,
        SPF_FIXED_ROUND_BASIS=False,
        SPF_DEBUG_GATE=False,
        DEBUG_TEXT_ENCODER_FORWARD_COUNT=False,
    )
    cfg = SimpleNamespace(
        INPUT=SimpleNamespace(SIZE=(32, 32)),
        MODEL=SimpleNamespace(BACKBONE=SimpleNamespace(NAME="fake"), INIT_WEIGHTS=""),
        TRAINER=SimpleNamespace(GL_SVDMSE=trainer_cfg),
        OPTIM=SimpleNamespace(
            NAME="adam",
            LR=1e-3,
            WEIGHT_DECAY=0.0,
            MOMENTUM=0.9,
            SGD_DAMPNING=0,
            SGD_NESTEROV=False,
            RMSPROP_ALPHA=0.99,
            ADAM_BETA1=0.9,
            ADAM_BETA2=0.999,
            STAGED_LR=False,
            NEW_LAYERS=(),
            BASE_LR_MULT=0.1,
            LR_SCHEDULER="single_step",
            STEPSIZE=(-1,),
            GAMMA=0.1,
            MAX_EPOCH=1,
            WARMUP_EPOCH=0,
            WARMUP_TYPE="constant",
            WARMUP_CONS_LR=1e-5,
            WARMUP_MIN_LR=1e-5,
            WARMUP_RECOUNT=True,
        ),
    )
    return cfg


class FakeTransformer(nn.Module):
    def forward(self, x):
        return x + x.mean(dim=0, keepdim=True)


class FakeVisual(nn.Module):
    def __init__(self, embed_dim, input_resolution=32):
        super().__init__()
        self.input_resolution = input_resolution
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(3, embed_dim, bias=False)

    def forward(self, image):
        x = self.pool(image).flatten(1)
        return self.proj(x)


class FakeClipModel(nn.Module):
    def __init__(self, embed_dim=16, vocab_size=49408, input_resolution=32):
        super().__init__()
        self.dtype = torch.float32
        self.visual = FakeVisual(embed_dim, input_resolution=input_resolution)
        self.transformer = FakeTransformer()
        self.positional_embedding = nn.Parameter(torch.zeros(77, embed_dim))
        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Parameter(torch.eye(embed_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)


def make_prompt_learner(
    use_spf=True,
    gamma_init=0.05,
    gamma_max=0.10,
    n_ctx=4,
    n_prompt=1,
    shared_init=False,
):
    torch.manual_seed(1)
    cfg = make_cfg(
        use_spf=use_spf,
        gamma_init=gamma_init,
        gamma_max=gamma_max,
        n_ctx=n_ctx,
        n_prompt=n_prompt,
        shared_init=shared_init,
    )
    classnames = ["alpha", "beta"]
    clip_model = FakeClipModel()
    prompt_learner = PromptLearner(cfg, classnames, clip_model)
    return cfg, classnames, clip_model, prompt_learner


def make_fake_trainer(prompt_learner, cfg):
    model = SimpleNamespace(prompt_learner=prompt_learner)
    return SimpleNamespace(model=model, cfg=cfg, optim=None, sched=None, _optims={}, _scheds={})


def assert_tensor_close(a, b, atol=1e-6, rtol=1e-6, msg=""):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(msg or f"Tensor mismatch:\n{a}\n!=\n{b}")


def test_gamma_initialization_and_range():
    cfg, _, _, prompt_learner = make_prompt_learner(use_spf=True, gamma_init=0.05, gamma_max=0.10)
    prompt_learner.refresh_spf_anchor()
    _, aux = prompt_learner.fuse_ctx_spf(prompt_learner.ctx_local, prompt_learner.ctx_global)
    gamma_mean = aux["gamma_mean"].item()
    gamma_std = aux["gamma_std"].item()
    gamma_min = aux["gamma_min"].item()
    gamma_max = aux["gamma_max"].item()

    assert abs(gamma_mean - cfg.TRAINER.GL_SVDMSE.SPF_GAMMA_BASE) <= 2e-3
    assert gamma_std > 0.0
    assert gamma_min >= cfg.TRAINER.GL_SVDMSE.SPF_GAMMA_MIN - 1e-8
    assert gamma_max <= cfg.TRAINER.GL_SVDMSE.SPF_GAMMA_MAX + 1e-8
    print("test_gamma_initialization_and_range passed")


def test_rank_gate_stays_in_shared_coordinates():
    cfg, _, _, prompt_learner = make_prompt_learner(use_spf=True, gamma_init=0.05, gamma_max=0.10)
    cfg.TRAINER.GL_SVDMSE.SPF_DEBUG_GATE = True
    prompt_learner.debug_gate = True
    prompt_learner.refresh_spf_anchor()
    _, aux = prompt_learner.fuse_ctx_spf(prompt_learner.ctx_local, prompt_learner.ctx_global)
    gamma = aux["gamma"]

    assert gamma.shape[-1] == int(aux["svd_rank"].item())
    assert gamma.shape[-1] <= prompt_learner.spf_max_rank
    assert aux["fused_ctx_dtype_is_fp32"].item() == 1.0
    assert aux["effective_dynamic_delta_fp32"].item() >= 0.0

    loss = aux["shared_pull_loss"] + gamma.mean()
    loss.backward()
    assert prompt_learner.rank_gate[-1].bias.grad is not None
    assert prompt_learner.rank_gate[-1].bias.grad.abs().sum().item() > 0
    assert prompt_learner.rank_bias.grad is not None
    assert prompt_learner.rank_bias.grad.abs().sum().item() > 0
    print("test_rank_gate_stays_in_shared_coordinates passed")


def test_bipolar_gate_and_push_loss_gradient_boundary():
    cfg, _, _, prompt_learner = make_prompt_learner(use_spf=True)
    cfg.TRAINER.GL_SVDMSE.SPF_BIPOLAR_GATE = True
    cfg.TRAINER.GL_SVDMSE.SPF_GAMMA_DELTA = 0.05
    cfg.TRAINER.GL_SVDMSE.SPF_PUSH_ALPHA = 10.0
    prompt_learner.spf_bipolar_gate = True
    prompt_learner.spf_gamma_delta = 0.05
    prompt_learner.spf_push_alpha = 10.0
    with torch.no_grad():
        prompt_learner.rank_bias.fill_(-1.0)
    prompt_learner.refresh_spf_anchor()

    _, aux = prompt_learner.fuse_ctx_spf(prompt_learner.ctx_local, prompt_learner.ctx_global)
    gamma = aux["rank_gamma"]

    assert gamma.min().item() < 0.0
    assert aux["negative_gamma_ratio"].item() > 0.0
    prompt_learner.ctx_global.grad = None
    prompt_learner.ctx_local.grad = None
    aux["push_loss"].backward()
    assert prompt_learner.ctx_global.grad is None or torch.allclose(
        prompt_learner.ctx_global.grad,
        torch.zeros_like(prompt_learner.ctx_global.grad),
    )
    assert prompt_learner.ctx_local.grad is not None
    print("test_bipolar_gate_and_push_loss_gradient_boundary passed")


def test_fp32_fused_prompt_reaches_text_encoder():
    cfg, classnames, clip_model, _ = make_prompt_learner(use_spf=True)
    cfg.TRAINER.GL_SVDMSE.SPF_DEBUG_GATE = True
    model = CustomCLIP(cfg, classnames, clip_model)
    model.train()
    model.prompt_learner.refresh_spf_anchor()
    image = torch.randn(2, 3, 32, 32)

    _, _, aux = model(image)

    assert aux["fused_ctx_dtype_is_fp32"].item() == 1.0
    assert model.text_encoder.forward_count == 0
    print("test_fp32_fused_prompt_reaches_text_encoder passed")


def test_basis_and_global_anchor_freeze_and_cache():
    _, _, _, prompt_learner = make_prompt_learner(use_spf=True)
    original_compute_shared_basis = sys.modules["trainers.GL_SVDMSE"].compute_shared_basis
    call_count = {"n": 0}

    def counting_compute_shared_basis(*args, **kwargs):
        call_count["n"] += 1
        return original_compute_shared_basis(*args, **kwargs)

    sys.modules["trainers.GL_SVDMSE"].compute_shared_basis = counting_compute_shared_basis
    try:
        prompt_learner.refresh_spf_anchor()
        assert call_count["n"] == 1
        basis_snapshot = prompt_learner._spf_basis.detach().clone()
        anchor_snapshot = prompt_learner._spf_global_anchor_coord.detach().clone()
        assert prompt_learner._spf_basis.requires_grad is False
        assert prompt_learner._spf_global_anchor_coord.requires_grad is False

        prompt_learner.fuse_ctx_spf(prompt_learner.ctx_local, prompt_learner.ctx_global)
        prompt_learner.fuse_ctx_spf(prompt_learner.ctx_local, prompt_learner.ctx_global)

        assert call_count["n"] == 1
        assert_tensor_close(prompt_learner._spf_basis, basis_snapshot)
        assert_tensor_close(prompt_learner._spf_global_anchor_coord, anchor_snapshot)
    finally:
        sys.modules["trainers.GL_SVDMSE"].compute_shared_basis = original_compute_shared_basis
    print("test_basis_and_global_anchor_freeze_and_cache passed")


def test_alignment_loss_and_ce_gradient_paths():
    cfg, classnames, clip_model, prompt_learner = make_prompt_learner(use_spf=True)
    prompt_learner.refresh_spf_anchor()

    prompt_learner.ctx_global.grad = None
    prompt_learner.ctx_local.grad = None
    _, aux = prompt_learner.fuse_ctx_spf(prompt_learner.ctx_local, prompt_learner.ctx_global)
    aux["shared_pull_loss"].backward()

    assert prompt_learner.ctx_global.grad is None or torch.allclose(
        prompt_learner.ctx_global.grad, torch.zeros_like(prompt_learner.ctx_global.grad)
    )

    prompt_learner.ctx_global.grad = None
    prompt_learner.ctx_local.grad = None
    model = CustomCLIP(cfg, classnames, clip_model)
    model.train()
    model.prompt_learner.refresh_spf_anchor()
    image = torch.randn(2, 3, 32, 32)
    label = torch.tensor([0, 1])
    logits, logits_global, _ = model(image)
    loss_ce = F.cross_entropy(logits, label)
    loss_ce.backward(retain_graph=True)

    assert model.prompt_learner.ctx_global.grad is None or torch.allclose(
        model.prompt_learner.ctx_global.grad,
        torch.zeros_like(model.prompt_learner.ctx_global.grad),
    )
    assert model.prompt_learner.ctx_local.grad is not None
    assert model.prompt_learner.ctx_local.grad.abs().sum().item() > 0

    model.prompt_learner.ctx_global.grad = None
    model.prompt_learner.ctx_local.grad = None
    loss_ce_global = F.cross_entropy(logits_global, label)
    loss_ce_global.backward()

    assert model.prompt_learner.ctx_global.grad is not None
    assert model.prompt_learner.ctx_global.grad.abs().sum().item() > 0
    print("test_alignment_loss_and_ce_gradient_paths passed")


def test_client_private_state_isolation():
    cfg, _, _, prompt_learner = make_prompt_learner(use_spf=True)
    fake_trainer = make_fake_trainer(prompt_learner, cfg)
    global_state = extract_spf_global_state(fake_trainer)
    private_a = clone_spf_private_state(prompt_learner)
    private_b = clone_spf_private_state(prompt_learner)

    mutated_a = {
        "ctx_local": private_a["ctx_local"].clone().add_(1.0),
        "rank_gate": {},
        "rank_bias": private_a["rank_bias"].clone().add_(1.0),
    }
    for key, value in private_a["rank_gate"].items():
        mutated_a["rank_gate"][key] = value.clone().add_(1.0)

    load_spf_client_state(fake_trainer, global_state, private_b)
    assert_tensor_close(prompt_learner.ctx_local.detach(), private_b["ctx_local"])
    for key, value in private_b["rank_gate"].items():
        assert_tensor_close(prompt_learner.rank_gate.state_dict()[key], value)
    assert_tensor_close(prompt_learner.rank_bias.detach().cpu(), private_b["rank_bias"])

    load_spf_client_state(fake_trainer, global_state, mutated_a)
    assert_tensor_close(prompt_learner.ctx_local.detach(), mutated_a["ctx_local"])
    for key, value in mutated_a["rank_gate"].items():
        assert_tensor_close(prompt_learner.rank_gate.state_dict()[key], value)
    assert_tensor_close(prompt_learner.rank_bias.detach().cpu(), mutated_a["rank_bias"])

    print("test_client_private_state_isolation passed")


def test_optimizer_state_isolation():
    cfg, _, _, prompt_learner = make_prompt_learner(use_spf=True)
    fake_trainer = make_fake_trainer(prompt_learner, cfg)

    rebuild_prompt_optimizer(fake_trainer)
    assert len(fake_trainer.optim.state) == 0
    param_ids = {
        id(param)
        for group in fake_trainer.optim.param_groups
        for param in group["params"]
    }
    gate_lr = cfg.OPTIM.LR * cfg.TRAINER.GL_SVDMSE.SPF_GATE_LR_MULT
    assert any(abs(group["lr"] - gate_lr) < 1e-12 for group in fake_trainer.optim.param_groups)
    for name, param in prompt_learner.rank_gate.named_parameters():
        assert id(param) in param_ids, f"rank_gate parameter missing from optimizer: {name}"
    assert id(prompt_learner.rank_bias) in param_ids

    loss = (
        prompt_learner.ctx_global.sum()
        + prompt_learner.ctx_local.sum()
        + sum(param.sum() for param in prompt_learner.rank_gate.parameters())
        + prompt_learner.rank_bias.sum()
    )
    loss.backward()
    fake_trainer.optim.step()
    assert len(fake_trainer.optim.state) > 0

    rebuild_prompt_optimizer(fake_trainer)
    assert len(fake_trainer.optim.state) == 0
    print("test_optimizer_state_isolation passed")


def test_fedavg_upload_safety():
    cfg, _, _, prompt_learner = make_prompt_learner(use_spf=True)
    fake_trainer = make_fake_trainer(prompt_learner, cfg)
    global_state = extract_spf_global_state(fake_trainer)

    assert set(global_state.keys()) == {"ctx_global"}
    assert "ctx_local" not in global_state
    assert "rank_gate" not in global_state
    assert "rank_bias" not in global_state
    print("test_fedavg_upload_safety passed")


def test_global_only_logits_ignore_private_state():
    cfg, classnames, clip_model, _ = make_prompt_learner(use_spf=True)
    model = CustomCLIP(cfg, classnames, clip_model)
    model.eval()
    image = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        base_logits = model(image, forward_mode="global_only")
        model.prompt_learner.ctx_local.copy_(torch.randn_like(model.prompt_learner.ctx_local) * 10.0)
        changed_local_logits = model(image, forward_mode="global_only")
        assert_tensor_close(base_logits, changed_local_logits, atol=0.0, rtol=0.0)

        model.prompt_learner.ctx_global.add_(torch.randn_like(model.prompt_learner.ctx_global))
        changed_global_logits = model(image, forward_mode="global_only")
        if torch.allclose(base_logits, changed_global_logits, atol=1e-6, rtol=1e-6):
            raise AssertionError("global_only logits did not change after ctx_global changed")

    print("test_global_only_logits_ignore_private_state passed")


def main():
    test_gamma_initialization_and_range()
    test_rank_gate_stays_in_shared_coordinates()
    test_bipolar_gate_and_push_loss_gradient_boundary()
    test_fp32_fused_prompt_reaches_text_encoder()
    test_basis_and_global_anchor_freeze_and_cache()
    test_alignment_loss_and_ce_gradient_paths()
    test_client_private_state_isolation()
    test_optimizer_state_isolation()
    test_fedavg_upload_safety()
    test_global_only_logits_ignore_private_state()
    print("ALL SPF TESTS PASSED")


if __name__ == "__main__":
    main()
