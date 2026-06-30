# Stable SPF-FedPHA: controlled replacement for the current SPF path

## What this patch changes

The original `spf` branch trains the fused prompt only:

```
L = CE(fused) + lambda_shared * MSE(P_shared(C_local), P_shared(C_global))
```

This is problematic because it has no direct supervision for the global prompt,
no independent local prediction objective, and changes the baseline's
null-space/private-preservation objective into a shared-space alignment loss.

The stable path uses:

```
L = CE(fused)
  + lambda_local * CE(local)
  + lambda_global * CE(global)
  + lambda_shared * MSE(P_anchor(C_local), P_anchor(C_anchor))
```

where `C_anchor` is the server `ctx_global` received by a client at the start
of its local update. It is frozen for that client's local update and retained
for that client's personalized evaluation in the same communication round.

## Why the frozen anchor is necessary

The current SPF code trains `C_local` against the pre-FedAvg global prompt but
evaluates it after replacing `ctx_global` with the post-FedAvg prompt. Since
fused inference explicitly depends on global directions, this changes the
model between training and testing. The patch preserves the training-time
anchor for PM-fused, while still evaluating GM using the newest FedAvg prompt.

## New diagnostics

`federated_spf_stable.py` writes `spf_stable_metrics.csv` with:

- `gm_micro`: strict global-only model using the new server prompt;
- `pm_fused_micro`: client local prompt plus its frozen fusion anchor;
- `pm_local_micro`: client local prompt alone;
- `fused_minus_local`: the only reliable answer to whether SPF fusion helps.

Do not claim fusion helps unless `fused_minus_local` is positive consistently
across seeds and not only at a single peak round.

## Recommended initial settings

### Office31
Small data and moderate domain shift; global object semantics are useful.

```
--spf_gamma_init 0.05 --spf_global_lambda 1.0 \
--spf_local_lambda 0.5 --spf_shared_lambda 0.0
```

### OfficeHome
Large domain/style shift. Avoid aggressively pulling local prompts into the
shared subspace.

```
--spf_gamma_init 0.02 --spf_global_lambda 0.5 \
--spf_local_lambda 0.75 --spf_shared_lambda 0.0
```

### DTD
Texture/domain cues are highly client-specific; use an almost local model.

```
--spf_gamma_init 0.01 --spf_global_lambda 0.5 \
--spf_local_lambda 1.0 --spf_shared_lambda 0.0
```

### Caltech101 / Food101 / OxfordPets / OxfordFlowers
Class-rich datasets benefit from a global classifier but need local safety.

```
--spf_gamma_init 0.03 --spf_global_lambda 1.0 \
--spf_local_lambda 0.5 --spf_shared_lambda 0.0
```

Only after the fixed-anchor result is stable should `spf_shared_lambda` be
swept over `{0.005, 0.01, 0.02}`. It should not start at `0.1`.
