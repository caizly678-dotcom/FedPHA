import torch
from torch.nn import functional as F


def choose_rank_by_energy(s, energy=0.90, min_rank=1, max_rank=None):
    s2 = s.float().pow(2)
    total = s2.sum().clamp_min(1e-12)
    ratio = torch.cumsum(s2, dim=0) / total
    threshold = torch.tensor(energy, device=s.device, dtype=ratio.dtype)
    r = int(torch.searchsorted(ratio, threshold).item()) + 1
    if max_rank is not None:
        r = min(r, int(max_rank))
    r = max(r, int(min_rank))
    r = min(r, s.numel())
    return r


def compute_shared_basis(ctx, energy=0.90, min_rank=1, max_rank=None):
    device = ctx.device
    dtype = ctx.dtype
    x = ctx.reshape(-1, ctx.shape[-1]).float()
    try:
        _, s, vh = torch.linalg.svd(x, full_matrices=False)
    except RuntimeError as err:
        print(f"SPF SVD failed on device, retrying on CPU: {err}")
        _, s, vh = torch.linalg.svd(x.cpu(), full_matrices=False)
        s = s.to(device)
        vh = vh.to(device)
    r = choose_rank_by_energy(s, energy=energy, min_rank=min_rank, max_rank=max_rank)
    basis = vh[:r, :].transpose(0, 1).contiguous()
    return basis.to(device=device, dtype=dtype), s.detach()


def compute_shared_basis_from_contexts(contexts, energy=0.90, min_rank=1, max_rank=None):
    if not contexts:
        raise ValueError("contexts must contain at least one prompt tensor")

    flattened = []
    for ctx in contexts:
        flattened.append(ctx.reshape(-1, ctx.shape[-1]))

    return compute_shared_basis(
        torch.cat(flattened, dim=0),
        energy=energy,
        min_rank=min_rank,
        max_rank=max_rank,
    )


def project_to_basis(ctx, basis):
    flat = ctx.reshape(-1, ctx.shape[-1])
    basis = basis.to(device=flat.device, dtype=flat.dtype)
    projected = flat @ basis @ basis.t()
    return projected.reshape_as(ctx)


def _identity_alignment(local_semantic, fallback_value):
    n_prompt, n_ctx, _ = local_semantic.shape
    eye = torch.eye(
        n_ctx,
        device=local_semantic.device,
        dtype=torch.float32,
    ).unsqueeze(0).expand(n_prompt, -1, -1).contiguous()
    confidence = torch.full(
        (n_prompt, n_ctx),
        float(fallback_value),
        device=local_semantic.device,
        dtype=torch.float32,
    )
    zero = torch.zeros((), device=local_semantic.device, dtype=torch.float32)
    one = torch.ones((), device=local_semantic.device, dtype=torch.float32)
    diagnostics = {
        "alignment_entropy": zero,
        "alignment_conf_mean": confidence.mean(),
        "alignment_conf_std": confidence.std(unbiased=False),
        "alignment_conf_min": confidence.min(),
        "alignment_conf_max": confidence.max(),
        "alignment_diagonal_cosine": zero,
        "alignment_matched_cosine": zero,
        "alignment_transport_peak": eye.amax(dim=-1).mean(),
        "alignment_fallback": one if fallback_value == 0 else zero,
    }
    return eye, confidence, diagnostics


@torch.no_grad()
def semantic_sinkhorn_alignment(
    local_semantic,
    global_semantic,
    tau=0.07,
    num_iters=5,
    eps=1e-8,
):
    """Compute token-level semantic soft alignment.

    Args:
        local_semantic: Tensor with shape [N, M, D].
        global_semantic: Tensor with shape [N, M, D].

    Returns:
        transport: Row-normalized alignment weights with shape [N, M, M].
        confidence: Per-local-token confidence with shape [N, M].
        diagnostics: Scalar float32 tensors for logging.
    """
    if local_semantic.shape != global_semantic.shape:
        raise ValueError(
            "local_semantic and global_semantic must have the same shape, "
            f"got {local_semantic.shape} and {global_semantic.shape}"
        )
    if local_semantic.dim() != 3:
        raise ValueError(
            "semantic_sinkhorn_alignment expects [N, M, D] tensors, "
            f"got dim={local_semantic.dim()}"
        )
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if num_iters < 1:
        raise ValueError(f"num_iters must be >= 1, got {num_iters}")

    local = local_semantic.detach().float()
    global_ = global_semantic.detach().float()
    n_prompt, n_ctx, _ = local.shape

    def fallback():
        return _identity_alignment(local, fallback_value=0.0)

    if not torch.isfinite(local).all() or not torch.isfinite(global_).all():
        return fallback()

    local_norm = F.normalize(local, p=2, dim=-1, eps=eps)
    global_norm = F.normalize(global_, p=2, dim=-1, eps=eps)
    similarity = torch.einsum("nid,njd->nij", local_norm, global_norm)
    if not torch.isfinite(similarity).all():
        return fallback()

    log_transport = similarity / float(tau)
    if not torch.isfinite(log_transport).all():
        return fallback()

    for _ in range(int(num_iters)):
        log_transport = log_transport - torch.logsumexp(
            log_transport, dim=-1, keepdim=True
        )
        log_transport = log_transport - torch.logsumexp(
            log_transport, dim=-2, keepdim=True
        )
    if not torch.isfinite(log_transport).all():
        return fallback()

    transport = log_transport.exp()
    transport = transport / transport.sum(dim=-1, keepdim=True).clamp_min(eps)
    if not torch.isfinite(transport).all():
        return fallback()

    if n_ctx > 1:
        entropy = -(transport * transport.clamp_min(eps).log()).sum(dim=-1)
        normalized_entropy = entropy / torch.log(
            torch.tensor(float(n_ctx), device=transport.device, dtype=torch.float32)
        )
        confidence = (1.0 - normalized_entropy).clamp(0.0, 1.0)
    else:
        normalized_entropy = torch.zeros(
            (n_prompt, n_ctx), device=transport.device, dtype=torch.float32
        )
        confidence = torch.ones(
            (n_prompt, n_ctx), device=transport.device, dtype=torch.float32
        )
    if not torch.isfinite(confidence).all():
        return fallback()

    diag_idx = torch.arange(n_ctx, device=transport.device)
    diagonal_cosine = similarity[:, diag_idx, diag_idx].mean()
    matched_cosine = (transport * similarity).sum(dim=-1).mean()
    diagnostics = {
        "alignment_entropy": normalized_entropy.mean().detach(),
        "alignment_conf_mean": confidence.mean().detach(),
        "alignment_conf_std": confidence.std(unbiased=False).detach(),
        "alignment_conf_min": confidence.min().detach(),
        "alignment_conf_max": confidence.max().detach(),
        "alignment_diagonal_cosine": diagonal_cosine.detach(),
        "alignment_matched_cosine": matched_cosine.detach(),
        "alignment_transport_peak": transport.amax(dim=-1).mean().detach(),
        "alignment_fallback": torch.zeros(
            (), device=transport.device, dtype=torch.float32
        ),
    }
    return transport.detach(), confidence.detach(), diagnostics


def align_global_context(global_ctx, transport):
    """Apply token alignment to global context.

    Args:
        global_ctx: Tensor with shape [N, M, D].
        transport: Tensor with shape [N, M, M].
    """
    if global_ctx.dim() != 3:
        raise ValueError(f"global_ctx must have shape [N, M, D], got {global_ctx.shape}")
    if transport.dim() != 3:
        raise ValueError(f"transport must have shape [N, M, M], got {transport.shape}")
    if transport.shape[0] != global_ctx.shape[0]:
        raise ValueError(
            f"transport N={transport.shape[0]} does not match global_ctx N={global_ctx.shape[0]}"
        )
    if transport.shape[1] != global_ctx.shape[1] or transport.shape[2] != global_ctx.shape[1]:
        raise ValueError(
            "transport must have shape [N, M, M] matching global_ctx, "
            f"got transport={transport.shape}, global_ctx={global_ctx.shape}"
        )
    return torch.einsum(
        "nij,njd->nid",
        transport.to(device=global_ctx.device, dtype=global_ctx.dtype),
        global_ctx,
    )
