# Method mapping and explicit completion choices

This repository implements the supplied manuscript's technical ideas independently. Equations refer to its Sections II–III. Unspecified choices below are assumptions, not recovered original settings.

## Equation-to-code map

| Paper | Code | Implementation |
| --- | --- | --- |
| (1)–(2) | `data.segment_features` | Aligned 3σ rejection; moving average on each selected segment |
| (3)–(4) | `model.DALSS.candidate` | k=3/7 convolutions, 1×1 fusion, affine instance normalization |
| (5)–(7) | `model.diagonal_zoh` | Stable diagonal A, positive dynamic step, exact ZOH |
| (8) | `model.HealthGate` | Health-weighted conditional discrepancy gate |
| (9) | `model.DALSS.forward` | Explicit gated temporal recurrence |
| (10)–(11) | `experiment.run_experiment` | Source MSE and gradient-reversal domain BCE |
| (12)–(13) | `model.DALSS.forward` | Mean temporal pooling and scalar linear head |
| (14) | `model.temporal_monotonicity` | ReLU of adjacent-window prediction increase beyond epsilon |
| (16) | `analysis.retrospective_groups` | 100-point two-channel standardized trajectories, KMeans and silhouette |
| (17)–(19) | `experiment.metrics` | RMSE, MAE in SOH fraction; MAPE in percent |

## Architecture

One state-space block; hidden dimension 64; two convolution branches with 32 channels each. There are no extra residual blocks or undocumented Mamba/S4 implementations. A is diagonal and parameterized as `-exp(log_decay)`; B and C are learned dense 64×64 bias-free projections. The per-channel step is `softplus(base_step + W_delta x)`, with initial base_step=-2 and decay=1. h0=0 for every window.

For a diagonal entry a<0 and step delta:

```
A_bar = exp(delta*a)
write = expm1(delta*a)/a * (B*x)
h_t = (1-g)*h_(t-1) + g*(A_bar*h_(t-1) + write_t)
```

This is the exact diagonal form of ZOH, not the first-order `delta*B*x` approximation. Effective diagonal retention lies between exp(delta*a) and 1. C maps the hidden state sequence before mean pooling. The paper alternates h/z notation at pooling; this implementation explicitly uses C. No clipping or monotone postprocessing is applied to predictions.

InstanceNorm uses each channel's temporal statistics **within the available input window** and has affine parameters but no running statistics. Symmetric convolutions see only that window, ending at the prediction time. The network is window-to-one, not a per-position causal forecasting decoder. Feature-level preprocessing uses only completed segments.

## Health-conditioned gate: details missing from the paper

- K=7 equally spaced references from 0.70 to 1.05; Gaussian soft assignments with temperature 0.05; kappa=0.5.
- An auxiliary linear head predicts health from the temporal mean of fused features **before** instance normalization. Its source MSE has weight 1.0. This extra loss and head location are explicit assumptions needed to train the otherwise unspecified auxiliary estimate.
- Window gate assignments use detached auxiliary estimates. Source conditional statistics use true source-training SOH; target statistics use auxiliary estimates only. Ground-truth target SOH is never consulted.
- For each reference, aggregate candidate writes over all positions of training/adaptation windows. Weighted first and second moments are streamed in float64. Each window contributes its q weight; overlapping windows consequently overlap in the statistics. This is a declared choice, distinct from the battery-uniform training sampler.
- Let mu_s/sigma_s and mu_t/sigma_t be weighted conditional moments. D=(abs(mu_s-mu_t)+abs(sigma_s-sigma_t))/s, where s is the source marginal channel standard deviation with floor 1e-4.
- S is the mean absolute change in source conditional means between adjacent supported references, divided by s. If none are supported, S=0.
- A reference is supported if each domain has weighted mass >=1 and effective window count `(sum q)^2/sum(q^2) >=8`. This count is a numerical support heuristic, not an independence claim about overlapping windows.
- g = 1 - kappa sum_k q_k m_k D_k/(1+D_k+S). It is shared over positions; 1-kappa <= g <=1. Unsupported references contribute zero, without renormalizing supported weights.
- First epoch starts at g=1. Recompute full-pool statistics at every epoch end using current weights, before source validation. The next epoch uses those buffers. Best weights and matching buffers are saved together; evaluation never recalibrates them.

The supplied text does not fix K, support thresholds, q/D/S formulas, auxiliary training, refresh schedule or detach behavior. Therefore gate behavior cannot be asserted to match the original implementation exactly.

## Loss signs and temporal pairing

The paper gives an adversarial minimax objective and a positive sum of losses. A naive positive BCE without gradient reversal would train a domain-specific encoder. This code reverses the representation gradient with coefficient lambda_domain=1.774; the domain classifier receives its ordinary BCE gradient. Numerically the optimized scalar is `task + domain + lambda_phy*TMR + lambda_aux*auxiliary`. Its gradients implement `task - lambda_domain*domain` for the representation and `domain` for the classifier, plus the respective regularizers. Logged BCE is unweighted.

TMR is not applied to shuffled batch neighbors. For each sampled window ending at cycle t, an extra window ending at t+1 from the same battery is evaluated. If no such numerical consecutive cycle exists, that pair is masked out. Valid source and target pairs contribute `mean(relu(y_next-y_current-0.005))`; source-only uses source pairs. Overlapping windows remain entirely within one role. This adds forward computation without reducing the specified 128+128 anchor batch.

## Experimental protocol

Explicit paper settings: total anchor batch 256, hidden dimension 64, learning rate 1e-3, Adam, weight decay 1e-4, maximum epochs 80, sequence length 128, epsilon=0.005, lambda_phy=0.703, lambda_domain=1.774.

Independent choices: window stride 1; seed 2026; 100 updates/epoch; early-stopping patience 15; deterministic flags; source validation uses pooled-window RMSE. Sorted battery IDs are shuffled with NumPy's seeded generator and divided into five source/four target folds. Source configuration i holds out fold i for test and (i+1)%5 for validation; other three train. Target configuration j holds out j; other three adapt. All 5×4 are run. The same battery is held out five times across the 20 target combinations, so these repetitions are dependent.

Raw per-segment cleaning is deterministic/local and can be cached before splitting because it fits no dataset-wide statistics. Dataset-wide standardization is fitted after battery assignment, exclusively on source training. Any upstream segment selection or feature generation that uses full-life SOH would violate this pipeline's assumptions.

Both window-pooled and battery-macro metrics are stored because the supplied paper does not fully disambiguate within-combination aggregation. Standard deviations across combinations use ddof=1. Clustering is a separate after-evaluation tool and never influences folds, normalization, model inputs or tuning.

## Reproduction boundary

Not recovered: original code, dataset quality criteria and excluded cell list, actual segment selection, capacity denominator, native file conversions, initial weights, exact architectures, original folds/seeds, Bayesian search space, baseline source code, private EV records, compiled deployment engine and Jetson measurement settings. The default model has 22,403 parameters including the discriminator, and 18,178 excluding it; no parameter-count equivalence is claimed. The 0.0673M count and reported accuracy/latency are not reproduced.

No full MIT/HUST training or baseline ranking is claimed. Strict numerical reproduction requires these missing materials. This project provides an executable, testable interpretation of the stated mechanism and protocol.
