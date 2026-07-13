import torch

from trainers.spf_utils import align_global_context, semantic_sinkhorn_alignment


def test_semantic_sinkhorn_alignment_shapes_and_ranges():
    local = torch.randn(2, 4, 8)
    global_ = torch.randn(2, 4, 8)

    transport, confidence, diagnostics = semantic_sinkhorn_alignment(
        local, global_, tau=0.07, num_iters=3
    )

    assert transport.shape == (2, 4, 4)
    assert confidence.shape == (2, 4)
    assert torch.allclose(transport.sum(dim=-1), torch.ones(2, 4), atol=1e-5)
    assert torch.isfinite(transport).all()
    assert torch.isfinite(confidence).all()
    assert torch.all(confidence >= 0)
    assert torch.all(confidence <= 1)
    assert diagnostics["alignment_fallback"].item() == 0

    aligned = align_global_context(global_, transport)
    assert aligned.shape == global_.shape


def test_semantic_sinkhorn_m_one_is_finite():
    local = torch.randn(3, 1, 5)
    global_ = torch.randn(3, 1, 5)

    transport, confidence, diagnostics = semantic_sinkhorn_alignment(local, global_)

    assert transport.shape == (3, 1, 1)
    assert torch.allclose(transport, torch.ones_like(transport))
    assert torch.allclose(confidence, torch.ones_like(confidence))
    assert torch.isfinite(diagnostics["alignment_entropy"])


def test_semantic_sinkhorn_nan_fallback():
    local = torch.randn(1, 3, 4)
    global_ = torch.randn(1, 3, 4)
    local[0, 0, 0] = float("nan")

    transport, confidence, diagnostics = semantic_sinkhorn_alignment(local, global_)

    expected = torch.eye(3).unsqueeze(0)
    assert torch.allclose(transport, expected)
    assert torch.allclose(confidence, torch.zeros_like(confidence))
    assert diagnostics["alignment_fallback"].item() == 1


def test_identical_tokens_prefer_diagonal():
    tokens = torch.randn(2, 5, 16)

    transport, _, diagnostics = semantic_sinkhorn_alignment(
        tokens, tokens, tau=0.05, num_iters=5
    )

    diagonal_mass = transport.diagonal(dim1=-2, dim2=-1).mean()
    off_diag_mass = (
        transport.sum() - transport.diagonal(dim1=-2, dim2=-1).sum()
    ) / (2 * 5 * 4)
    assert diagnostics["alignment_matched_cosine"] >= (
        diagnostics["alignment_diagonal_cosine"] - 1e-4
    )
    assert diagonal_mass > off_diag_mass


def test_detach_private_blocks_private_gradient():
    local = torch.randn(2, 3, 4, requires_grad=True)
    global_shared = torch.randn(2, 3, 4)
    basis = torch.eye(4)[:, :2]
    flat = local.reshape(-1, 4)
    local_shared = (flat @ basis @ basis.t()).reshape_as(local)
    local_private = local - local_shared
    gamma = torch.full((2, 3, 1), 0.3)

    fused = (
        local_private.detach()
        + (1 - gamma) * local_shared
        + gamma * global_shared.detach()
    )
    fused.pow(2).sum().backward()

    grad = local.grad.reshape(-1, 4)
    shared_grad = grad @ basis @ basis.t()
    private_grad = grad - shared_grad
    ratio = private_grad.norm() / grad.norm().clamp_min(1e-12)
    assert ratio.item() < 1e-6


def test_identity_alignment_matches_original_residual_formula():
    local = torch.randn(2, 4, 6)
    global_ = torch.randn(2, 4, 6)
    basis = torch.eye(6)[:, :3]
    gamma = torch.full((2, 4, 1), 0.2)

    flat_local = local.reshape(-1, 6)
    flat_global = global_.reshape(-1, 6)
    local_shared = (flat_local @ basis @ basis.t()).reshape_as(local)
    global_shared = (flat_global @ basis @ basis.t()).reshape_as(global_)
    local_private = local - local_shared

    original = local + gamma * (global_shared.detach() - local_shared)
    isolated = (
        local_private.detach()
        + (1 - gamma) * local_shared
        + gamma * global_shared.detach()
    )

    assert torch.allclose(original, isolated, atol=1e-6)
