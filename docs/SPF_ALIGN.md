# Alignment-aware SPF

## Problem in spf1

The current SPF fusion assumes context token position `i` in the Local prompt should fuse with position `i` in the Global prompt. Local and Global context tokens are learned independently on different objectives, so equal positions do not necessarily carry the same semantics.

## Flow

1. Contextual Token Encoding: encode Local and Global prompts with the CLIP text transformer and average each context token state over classes, giving `L, G in R^{N x M x D}`.
2. Sinkhorn Soft Alignment: compute cosine similarity and a row-normalized transport matrix `A in R^{N x M x M}` with log-space Sinkhorn.
3. Shared Subspace Projection: align Global context first, then compute the SPF shared basis from aligned Global context.
4. Confidence-controlled Residual Fusion: use alignment entropy to down-weight uncertain token matches.
5. Private Gradient Isolation: optionally detach the Local private subspace in the fused loss path.

## Core Formulas

Cosine similarity:

```text
S_ij = cos(L_i, G_j)
```

Sinkhorn alignment:

```text
log A = Sinkhorn(S / tau)
aligned_G_i = sum_j A_ij G_j
```

Confidence from normalized entropy:

```text
H_i = -sum_j A_ij log(A_ij)
c_i = clamp(1 - H_i / log(M), 0, 1)
gamma_i = gamma_base * spf_scale * c_i^conf_power
```

SPF residual fusion:

```text
L_s = P_B(L)
G_s = P_B(aligned_G)
L_p = L - L_s
fused = L + gamma * (stopgrad(G_s) - L_s)
```

With private isolation:

```text
fused = stopgrad(L_p) + (1 - gamma) * L_s + gamma * stopgrad(G_s)
```

These two forms are forward-equivalent. The isolated form prevents the fused loss from updating the Local private subspace.

## New Arguments

`--spf_use_alignment`: enable semantic Sinkhorn alignment before SPF fusion.

`--spf_align_tau`: alignment temperature, default `0.07`, must be greater than `0`.

`--spf_sinkhorn_iters`: number of log-space Sinkhorn iterations, default `5`, must be at least `1`.

`--spf_conf_power`: exponent applied to alignment confidence, default `1.0`, must be non-negative.

`--spf_detach_private`: detach the Local private subspace in the fused loss path.

## Ablations

A1 current SPF:

```bash
python federated_main.py --trainer GL_SVDMSE --dataset caltech101 --num_shots 16 --backbone "ViT-B/16" --num_users 10 --seed 1 --use_spf
```

A2 semantic alignment only:

```bash
python federated_main.py --trainer GL_SVDMSE --dataset caltech101 --num_shots 16 --backbone "ViT-B/16" --num_users 10 --seed 1 --use_spf --spf_use_alignment
```

A3 full method:

```bash
python federated_main.py --trainer GL_SVDMSE --dataset caltech101 --num_shots 16 --backbone "ViT-B/16" --num_users 10 --seed 1 --use_spf --spf_use_alignment --spf_detach_private
```

## Properties

The method adds no trainable parameters and no federated communication. Alignment is computed locally on each client. The server still aggregates only `prompt_learner.ctx_global`, while `prompt_learner.ctx_local` remains private.

The original spf1 behavior is the special case `A=I`, `confidence=1`, and `detach_private=False`.

## Metrics

`matched cosine`: average cosine similarity after transport-weighted matching.

`diagonal cosine`: average cosine similarity for same-position token pairs.

`alignment entropy`: normalized entropy of each alignment row, averaged over tokens.

`correction ratio`: `||fused_ctx - ctx_local||_F / max(||ctx_local||_F, 1e-12)`.

`oracle accuracy`: percentage of samples where Local or Global predicts correctly.

`global-help-local-wrong rate`: percentage of samples where Global is correct and Local is wrong.
