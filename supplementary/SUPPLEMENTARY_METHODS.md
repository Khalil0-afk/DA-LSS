# Supplementary Methods and Reproducibility Protocol

DA-LSS open-source implementation, version 0.3.0

This supplement adapts the definitions and equations in the author-provided revision response into English, retaining their physical scope and assumptions. Page numbers below refer to the 49-page response, not the manuscript. The supplied response and manuscript are source documents; their claims are not treated as measured results of this software release. Source-document hashes and unresolved inconsistencies are recorded in `docs/response_commitments.md`.

## S1. Measurement level and electrical signals

The real-vehicle evaluation object is the individual cell. The voltage input is the terminal voltage of the evaluated cell, while the current input is the pack main-circuit current. As stated in the response, this current can approximate the evaluated cell current only for a series-connected cell when parallel current diversion is absent and balancing current is negligible:

$$
I_j(t) \approx I_{\mathrm{pack}}(t).
$$

This approximation concerns the physical meaning of the current input. It does not imply that series-connected cells have equal usable capacities or equal SOH. Voltage, current and temperature originate from their respective acquisition circuits and sensors, are collected by the BMS, and are retrieved through CAN records. CAN identifiers, bit layouts, scale factors and offsets are vehicle-specific and must come from the authorized acquisition specification. The repository's byte-aligned mapping example is synthetic and is not an OEM DBC. No real CAN calibration is inferred from the response. [Response pp. 2-3, 20-21, 37-38.]

The provided pipeline decodes declared signals into engineering units, associates records by vehicle/cell identity, and matches cell-voltage records with same-vehicle pack-current observations at or before the voltage timestamp. A user-specified maximum age prevents stale matches. A future current observation is never borrowed. Prediction-to-reference association is a separate post-inference operation, also constrained by vehicle, cell, timestamp and maximum age.

## S2. Charge integration, usable capacity and SOH

For the evaluated cell j over a continuous segment with defined endpoints, the signed charge change is

$$
\Delta Q_j = \frac{1}{3600}\int_{t_a}^{t_b} I_j(t)\,\mathrm{d}t.
$$

Current is in amperes, time in seconds, and charge in ampere-hours. Under reliable cell-level endpoint SOC and approximately unit coulombic efficiency, usable capacity is estimated as

$$
Q_{j,\mathrm{use}} \approx \frac{\Delta Q_j}{\mathrm{SOC}_j(t_b)-\mathrm{SOC}_j(t_a)}.
$$

SOC is expressed on a zero-to-one scale. For a segment spanning the complete usable SOC range, the denominator has magnitude one. For a partial segment, its charge integral cannot be identified with full usable capacity without accounting for the SOC excursion. Charging current and increasing SOC are positive; a discharge segment has both negative charge change and negative SOC change, yielding positive capacity. The implementation rejects mixed current polarity, inconsistent signs, insufficient SOC excursion, non-increasing timestamps and unsupported integration gaps.

The cell SOH definition is

$$
\mathrm{SOH}_j = \frac{Q_{j,\mathrm{use}}}{Q_{j,\mathrm{nom}}}.
$$

For pack-level definitions, the response separately gives

$$
\Delta Q_{\mathrm{pack}} = \frac{1}{3600}\int_{t_a}^{t_b} I_{\mathrm{pack}}(t)\,\mathrm{d}t,
$$

$$
Q_{\mathrm{pack,use}} \approx \frac{\Delta Q_{\mathrm{pack}}}{\mathrm{SOC}_{\mathrm{pack}}(t_b)-\mathrm{SOC}_{\mathrm{pack}}(t_a)},
$$

$$
\mathrm{SOH}_{\mathrm{pack}}=\frac{Q_{\mathrm{pack,use}}}{Q_{\mathrm{pack,nom}}}.
$$

Pack definitions must not be used to assign one pack SOH value to every cell. The integration-derived estimate returned by this repository is labeled separately from the BMS-reported reference used for evaluation. The response explicitly states that reported cell-level errors quantify agreement with corresponding BMS estimates and do not independently establish accuracy against measured cell capacity. Weak-cell identification and within-pack fault localization have not been independently validated. [Response pp. 2-3, 5-6, 37-38.]

## S3. Segment processing and feature definitions

The completed measurement segment is the basic feature unit. Voltage and current are cleaned separately using the three-standard-deviation criterion:

$$
|x_t-\mu|>3\sigma.
$$

The implementation combines the two rejection masks to retain voltage/current alignment. Nonfinite paired rows are removed first. A trailing moving average is applied independently within each cleaned segment:

$$
x'_t=\frac{1}{W}\sum_{r=0}^{W-1}x_{t-r}.
$$

The response's prose specifies backward/trailing averaging; some pasted older equations still show a forward index. This supplement uses the clarified trailing direction. The current implementation chooses W=15, valid boundaries and no padding. These settings are explicit completion choices, not recovered original experimental metadata.

For n retained, smoothed observations, mean voltage, mean current, voltage entropy and population voltage standard deviation are

$$
\mu_V=\frac{1}{n}\sum_{r=1}^{n}V'_r,\qquad \mu_I=\frac{1}{n}\sum_{r=1}^{n}I'_r,
$$

$$
H_V=-\sum_{b:p_b>0}p_b\log p_b,\qquad \sigma_V=\sqrt{\frac{1}{n}\sum_{r=1}^{n}(V'_r-\mu_V)^2}.
$$

The voltage histogram uses 32 fixed bins over [2.0,4.5] V and natural logarithms. These entropy settings are declared implementation choices because the supplied documents do not uniquely specify them. Feature units are V, A, nat and V, in that order. Out-of-range smoothed voltages and insufficient segments raise errors rather than being padded or silently discarded. Each completed segment contributes one chronologically ordered vector; 128 vectors form an input window, whose output corresponds to the final segment. Data after that prediction time are not part of the input. [Response R1.7, pp. 27-29.]

## S4. Health-conditioned state propagation

The continuous state-space definition and input-dependent step are

$$
\dot h(t)=Ah(t)+Bx(t),\qquad z(t)=Ch(t),
$$

$$
\Delta_t=\mathrm{Softplus}(\mathrm{Parameter}+\mathrm{Proj}(x_t)).
$$

Zero-order hold produces the base discrete operators

$$
\bar A_t=\exp(\Delta_t A),
$$

$$
\bar B_t=(\Delta_t A)^{-1}\bigl(\exp(\Delta_t A)-I\bigr)\Delta_t B.
$$

The reference implementation chooses a negative diagonal A and uses expm1 to evaluate the equivalent scalar expression accurately for small steps. B and C are dense projections. These structural details are implementation choices; the original layer specification was not provided.

The candidate write contribution is

$$
u_{i,t}=\bar B_{i,t}x_{i,t}.
$$

Let q_ik denote nonnegative health-condition weights assigned from an auxiliary SOH estimate to K health reference points:

$$
q_{i,k}\geq0,\qquad \sum_{k=1}^{K}q_{i,k}=1.
$$

D_kj measures conditional source-target differences in candidate-write means and standard deviations; S_j is a source-domain statistical proxy for health relevance; m_k is one only when both domains have sufficient support. The update weight given in the response is

$$
g_{i,j}=1-\kappa\sum_{k=1}^{K}q_{i,k}m_k\frac{D_{k,j}}{1+D_{k,j}+S_j},\qquad 0\leq\kappa<1.
$$

Defining the diagonal matrix G_i, the state update and its equivalent effective operators are

$$
G_i=\mathrm{diag}(g_i),
$$

$$
h_{i,t}=(I-G_i)h_{i,t-1}+G_i\bigl(\bar A_{i,t}h_{i,t-1}+\bar B_{i,t}x_{i,t}\bigr),
$$

$$
\bar A^*_{i,t}=I-G_i+G_i\bar A_{i,t},\qquad \bar B^*_{i,t}=G_i\bar B_{i,t}.
$$

The effective operators are algebraically equivalent descriptions of the gated recurrence, not additional independent trainable matrices. The gate is shared across sequence positions in the same window. Larger conditional discrepancies reduce update weights, while stronger source-domain health relevance mitigates that reduction. Unsupported health conditions do not contribute; target true SOH is not used. [Response R1.1, pp. 13-14.]

The response does not fully specify q, D, S, support thresholds or auxiliary training. This implementation uses seven references in [0.70,1.05], Gaussian soft weights with temperature 0.05, kappa 0.5, effective support at least eight and mass at least one in both domains. Conditional discrepancy is the absolute mean difference plus absolute standard-deviation difference divided by source marginal channel standard deviation. Relevance is the mean adjacent-reference change in supported source conditional means under the same scaling. The auxiliary head is source-supervised, assignments are detached, and statistics are recalibrated from source training plus unlabeled adaptation windows at epoch boundaries. These choices are fully stated in `docs/method.md` and must not be attributed to unprovided original code.

## S5. Adversarial alignment and tolerance-based TMR

Source supervision is mean-squared error; domain BCE uses a gradient reversal layer so that the classifier minimizes domain loss while the representation maximizes it. The coefficient lambda_dom acts on the reversed representation gradient. Positive scalar BCE alone, without reversal, would not implement the stated adversarial objective.

The tolerance-based regularizer in the response is

$$
\mathcal L_{\mathrm{phy}}=\frac{1}{n-1}\sum_{i=1}^{n-1}\max\bigl(0,\hat y_{i+1}-\hat y_i-\epsilon\bigr),\qquad \epsilon\geq0.
$$

It penalizes only the excess above the allowable local increase and does not guarantee strict monotonicity. Code pairs consecutive endpoints of windows from the same battery, never arbitrary neighbors in a shuffled minibatch. Missing consecutive endpoints are masked. The sensitivity suite retrains five settings under the same seed and partition: (lambda_phy,epsilon) equal to (0.703,0.005), (0.25,0.005), (1.25,0.005), (0.703,0.001) and (0.703,0.025). Its displayed target battery is selected before training or inspecting predictions. [Response R1.5, pp. 24-25.]

## S6. Partitioning, fitting and retrospective grouping

For each MIT-to-HUST or HUST-to-MIT task, the source has five battery folds: three supervised training folds, one validation fold and one in-domain test fold. The target has four folds: three unlabeled-adaptation folds and one held-out transfer-test fold. All five source configurations are crossed with all four target configurations, producing 20 combinations per direction. Battery identity is assigned before normalization and window construction. All windows from a battery keep one role within a combination.

Dataset-level preprocessing is fitted only on source training. Gate statistics use source training and unlabeled target adaptation. Source validation alone controls tuning, early stopping and checkpoint selection. Source/target test references are evaluated after predictions are saved. The minibatch contains 128 source and 128 target anchor windows, with each pool sampled by uniformly choosing a battery and then one valid window, both with replacement. Manifests publish identity assignments, battery/window counts, actual adaptation fractions, seeds, strides, configurations and data hashes. [Response R1.7, pp. 27-29; R2.2-3, pp. 32-33; R3.5-6, pp. 44-46.]

Complete reference trajectories are used only after prediction finalization for retrospective K-means grouping. Each battery is resampled to 100 points in normalized cycle progression; SOH and actual cycle coordinates are independently standardized with pooled channel means and standard deviations and concatenated. Euclidean silhouette selects K from 2 through 10. For three clusters, names FAS/TAS/SAS are assigned by observed degradation rate. Neither membership nor trajectory-derived statistics enter selection, fitting or inference. The silhouette expression is

$$
S=\frac{1}{N}\sum_{i=1}^{N}\frac{b_i-a_i}{\max(a_i,b_i)}.
$$

Partition standard deviations describe variation across overlapping partitions; they are not confidence intervals from independent repeats.

## S7. Two distinct vehicle protocols

The source documents contain conflicting versions. The current manuscript and several retained response passages describe evaluation of a public-data-trained frozen model on entirely unseen vehicles. A newer substantive reply, R1.4 on pp. 22-23, instead specifies pre-deployment unsupervised adaptation. Both protocols are available and labeled separately; their results must not be pooled under one claim.

The new adaptation protocol initializes from a public-data checkpoint selected on source validation. Four adaptation vehicles are identity-disjoint from 20 reserved test vehicles. Exactly 32 distinct stride-32 windows per adaptation vehicle form 128 available windows. Source labels continue to provide regression supervision; EV labels are removed. Adam uses learning rate 1e-4 and weight decay 1e-4, for ten epochs of four replacement-sampled updates: 40 total updates. The epoch is a fixed update budget, not a complete pass through every window. Checkpoint selection uses source validation only; all ten epochs are completed. Model/scaler/gate state is frozen before deployment; the discriminator and gradient reversal are inactive during inference.

The implementation retains the pretrained source scaler, source auxiliary supervision and source-pair TMR. It does not construct additional unregistered target successor windows outside the declared adaptation pool. Those details and a per-domain batch of 128 are completion choices because the response does not specify them. Adaptation manifests retain reserved identities, selected window endpoints and initial-checkpoint provenance. A test-cell name reused on another vehicle never merges the two FIFO histories. No reserved EV test records are loaded during adaptation.

## S8. Online FIFO and amortized computation

The buffer stores 128 valid vectors for one evaluated cell in one vehicle. No neural estimate is generated before the buffer is complete; there is no padding or extrapolation. Subsequent predictions run at batch size one after every 32 new valid vectors. Neighboring windows share 96 vectors, giving

$$
r_{\mathrm{overlap}}=\frac{128-32}{128}\times100\%=75\%.
$$

Invalid vectors do not advance the update count. Timestamp order is enforced per vehicle and cell. An existing BMS/initial health value may be displayed externally during cold start, as discussed in the response; it is not produced by the neural estimator and is not scored as a model prediction.

With C_window operations per forward pass, the steady-state amortized model cost per new valid sequence point is

$$
C_{\mathrm{point}}=\frac{C_{\mathrm{window}}}{32}.
$$

This excludes continuously incurred acquisition and preprocessing work. [Response R3.3, p. 41.]

## S9. Deployment cost definitions and measurement scopes

The response defines inference parameter count and actual exported file size using decimal millions:

$$
N_{\mathrm{params}}=\frac{1}{10^6}\sum_{p\in\mathcal P_{\mathrm{infer}}}\mathrm{numel}(p),\qquad S_{\mathrm{model}}=\frac{B_{\mathrm{file}}}{10^6}.
$$

Shared parameter tensors are counted once. The deployment discriminator is excluded. The file metric includes serialized constants, buffers and metadata. This implementation's architecture does not reproduce the response's reported 0.0673M parameter figure.

The operation-count convention is

$$
C_{\mathrm{MFLOPs}}=\frac{2N_{\mathrm{MAC}}+N_{\mathrm{other}}}{10^6}.
$$

The resource script counts actual Conv1d and Linear calls, the gate matrix product, and explicitly listed scalar arithmetic. Normalization and transcendental implementation costs are not fully assigned, so the output is named accounted MFLOPs and includes an unaccounted-operator list. It is not asserted to reproduce the paper's full MFLOPs column or a hardware instruction count.

Forward-only and end-to-end latency use separate timing boundaries:

$$
\tau_{\mathrm{infer}}=\frac{10^3}{R}\sum_{r=1}^{R}(t_{\mathrm{forward,end},r}-t_{\mathrm{forward,start},r}),
$$

$$
\tau_{\mathrm{e2e}}=\frac{10^3}{R}\sum_{r=1}^{R}(t_{\mathrm{output},r}-t_{\mathrm{input},r}).
$$

The response's end-to-end definition starts when required records are available and excludes waiting to acquire a full window. The supplied harness can include completed-segment cleaning, feature extraction, window organization, normalization, transfer, forward inference and scalar output; CAN acquisition/decoding and asynchronous matching are outside that harness and explicitly reported as excluded. CUDA execution is synchronized. The default is ten warmups and 100 measured repetitions, float32, batch one, length128.

The response's runtime-memory definition is

$$
M_{\mathrm{peak}}=\frac{1}{10^6}\max_{t\in\mathcal T_{\mathrm{eval}}}M_{\mathrm{runtime}}(t).
$$

This repository samples process RSS every millisecond, including runtime overhead; that sample maximum is labeled as such rather than claimed to be an exact instantaneous maximum. A CUDA allocator peak is reported separately. Interfaces and scopes must be identical across a hardware comparison.

For a user-supplied post-warmup platform-power trace, the response defines

$$
\bar P_{\mathrm{platform}}=\frac{1}{T}\int_{t_a}^{t_b}P_{\mathrm{platform}}(t)\,\mathrm{d}t,\qquad T=t_b-t_a,
$$

$$
E_{\mathrm{window}}=\frac{10^3}{R_P}\int_{t_a}^{t_b}P_{\mathrm{platform}}(t)\,\mathrm{d}t=\frac{10^3\bar P_{\mathrm{platform}}T}{R_P}.
$$

Power is watts and per-window energy is millijoules. Samples are integrated with their actual timestamps using trapezoidal integration; idle/base power is not subtracted. R_P must be the number of complete processing windows within the same measurement interval. No synthetic watts are substituted for unavailable hardware telemetry. [Response R3.2, pp. 38-40.]

## S10. Reporting and unresolved author assets

Vehicle errors are calculated independently and summarized across vehicles with a mean and sample standard deviation. Charging-protocol results and vehicle-by-protocol results are also saved. Protocol membership is attributed to the prediction endpoint; a window containing mixed history is not silently called protocol-pure. A BMS reference is matched to the same evaluated cell and timestamp and is read only after the prediction is formed.

The original public-data archives, physical-cell identity corrections, actual four/20-vehicle identities and records, OEM CAN specification, original complete architecture and all external baseline implementations were not supplied. The response also names C11 in its vehicle table while manuscript passages name A05; it retains contradictory direct-frozen and adapted-vehicle descriptions and S4/Mamba versus FESD/MINE-Graph-Mamba descriptions. These are recorded for author reconciliation, not silently resolved with fabricated data or results. External baselines, original reported error values and Jetson measurements are not regenerated from manuscript tables. The release provides executable mechanisms and explicit validation evidence, not a retrospective certification of those empirical claims.
