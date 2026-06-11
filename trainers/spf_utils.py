import torch


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


def project_to_basis(ctx, basis):
    flat = ctx.reshape(-1, ctx.shape[-1])
    basis = basis.to(device=flat.device, dtype=flat.dtype)
    projected = flat @ basis @ basis.t()
    return projected.reshape_as(ctx)
