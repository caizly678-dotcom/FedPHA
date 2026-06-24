# Lambda Sweep Summary

## Table 1: Performance Results

| Dataset | Lambda | Last10 Avg | Last10 Std | Best | Best Round | Final | Δ vs λ=0.1 | Δ vs λ=0 | Result Path |
|---|---|---|---|---|---|---|---|---|---|
| oxford_pets | 0 | 99.3885 | 0.0852 | 99.5967 | 37 | 99.2684 | 0.1562 | 0.0000 | output/oxford_pets/GL_SVDMSE/shot_16/beta_0.5/ep1_r50/alpha1.0_ratio0.8/seed_42/spf_g0.01_e0.8_r8_slambda0.0/para1 |
| oxford_pets | 0.03 | 99.3411 | 0.1414 | 99.4958 | 41 | 99.3429 | 0.1087 | -0.0474 | output/oxford_pets/GL_SVDMSE/shot_16/beta_0.5/ep1_r50/alpha1.0_ratio0.8/seed_42/spf_g0.01_e0.8_r8_slambda0.03/para1 |
| oxford_pets | 0.1 | 99.2324 | 0.1645 | 99.5213 | 38 | 99.2681 | 0.0000 | -0.1562 | output/oxford_pets/GL_SVDMSE/shot_16/beta_0.5/ep1_r50/alpha1.0_ratio0.8/seed_42/spf_g0.01_e0.8_r8_slambda0.1/para1 |
| oxford_pets | 0.3 | 99.3917 | 0.1008 | 99.5716 | 37 | 99.4205 | 0.1593 | 0.0032 | output/oxford_pets/GL_SVDMSE/shot_16/beta_0.5/ep1_r50/alpha1.0_ratio0.8/seed_42/spf_g0.01_e0.8_r8_slambda0.3/para1 |
| oxford_pets | 1.0 | 99.3209 | 0.1419 | 99.5457 | 34 | 99.3940 | 0.0885 | -0.0677 | output/oxford_pets/GL_SVDMSE/shot_16/beta_0.5/ep1_r50/alpha1.0_ratio0.8/seed_42/spf_g0.01_e0.8_r8_slambda1.0/para1 |

## Table 2: Mechanism Diagnostics

| Dataset | Lambda | Mean CE Loss | Mean MSE Loss | Mean λ×MSE | Mean Effective Pull Ratio | Mean Utility | Positive Utility Ratio | Negative Utility Ratio | Log Path |
|---|---|---|---|---|---|---|---|---|---|
| oxford_pets | 0 |  |  |  |  |  |  |  |  |
| oxford_pets | 0.03 |  |  |  |  |  |  |  |  |
| oxford_pets | 0.1 |  |  |  |  |  |  |  |  |
| oxford_pets | 0.3 |  |  |  |  |  |  |  |  |
| oxford_pets | 1.0 |  |  |  |  |  |  |  |  |
