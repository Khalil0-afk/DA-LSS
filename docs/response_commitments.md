# Revision-response commitments and implementation audit

Source response: `Revision notes_6_22.pdf`, 49 pages, SHA-256 `f87a0a551f4148af7ca6a792ae3086bba6f9fa71fcfe91123841ab0b20c5b605`. Source manuscript: `revised.pdf`, 13 pages, SHA-256 `47ef737600afe4896b21216eaf2943ca30e6dcb83e92419c3e85c5621eeddb9f`. The original documents are excluded from the release. This is a code/supplement audit, not a replacement response letter or an assertion that all manuscript experiments have been independently reproduced.

| Response location | Commitment | Release artifact | Evidence/status |
| --- | --- | --- | --- |
| Area Editor pp.2-3; R3.1 pp.37-38 | Explain cell voltage, pack current and series-current approximation | Supplement S1-S2; `vehicle_data.py` | Definitions retained in English; physical assumptions explicit |
| Area Editor p.3; R3.1 p.38 | Document acquisition, CAN conversion and time matching | `decode-can`, `match-signals`, `matched-segments`; `docs/vehicle_data.md` | Synthetic mapping and causal-match tests pass; actual OEM mapping required |
| Same | Capacity integration, partial-SOC conversion and SOH definitions | `capacity-soh`; Supplement S2 | Charge/discharge, sign, gap and SOC-excursion tests pass |
| Same | Match the BMS reference to the evaluated cell and prediction time | `match-references`; `replay` | Pack broadcasts and future-reference matches rejected; references excluded from inference |
| R1.1 pp.13-14 | Gate variables and effective transition/input operators | `model.py`; Supplement S4 | Original gate/effective-operator equations retained; unspecified q/D/S choices distinguished |
| R1.5 pp.24-25 | Tolerance-based TMR and controlled parameter cases | `suite --suite sensitivity`; Supplement S5 | Five settings, same seed/split, preregistered display battery |
| R1.7 pp.27-29 | Explicit features, units, preprocessing, smoothing and boundaries | `preprocessing.json`, `prepare`, `data.py`; Supplement S3 | Trailing valid smoothing, fixed entropy bins, provenance and numeric cycle-order test |
| R1.7; R2.2-3 pp.32-33 | Five source/four target folds and all eligible test batteries | `crossval`; per-run manifests | 40 synthetic combinations run; actual study assignments not supplied |
| R1.7 | Publish identities, valid-window counts, proportions, stride, seed | `manifest.json` per actual run | Implemented for generated/real user inputs; no invented original battery list |
| R1.7 | Bayesian search space/budget/seed/early-stop configuration | `tune`; `tuning_protocol.json`; `search_protocol.json` | Source-validation-only tests pass; space/budget declared implementation choices |
| R1.4 pp.22-23 | Pretrained initialization, four adaptation vehicles separate from 20 tests | `adapt-vehicle`; `vehicle_adaptation.json` | Vehicle-level overlap rejected; test measurements not loaded |
| R1.4 Table R p.23 | 32 windows per adaptation vehicle, stride32, Adam1e-4/wd1e-4, 10 epochs/40 updates | `vehicle_adaptation.py` | Exact default budget exercised on synthetic vehicle inputs; source-validation selection |
| R1.4 | EV SOH labels excluded from adaptation, tuning and selection | adaptation loader strips labels | Label perturbation gives identical adapted model state |
| R3.3 p.41 | FIFO128/stride32/75% overlap, batch1 and unpadded cold start | `OnlineEstimator`, `replay`; Supplement S8 | Validity/stride/cold-start tests, including reused cell IDs across vehicles |
| R3.3 | Per-new-point computation is per-window computation divided by32 | `resources.py` | Exact arithmetic identity tested |
| R3.2 pp.38-40 | Inference parameters, actual model bytes, MAC/FLOP convention | `resources`; Supplement S9 | Implemented with explicit uncounted normalization/transcendental costs; accounted MFLOPs only |
| R3.2 | Warmup/repetitions, forward/e2e boundaries and runtime-memory scope | `resources` | Measured on local hardware; sampled RSS and CUDA allocator separately labeled |
| R3.2 | Platform-power integration and energy per completed window | `power-energy`; Supplement S9 | Nonuniform-time integration test; actual telemetry still required |
| R2.6 pp.36-37 | Errors by vehicle and charging mode | `replay` metrics | Vehicle/protocol/vehicle-by-protocol grouping tested |
| R3.6 pp.44-46 | Full-trajectory clustering only after prediction finalization | `cluster`, `group-score`, `group-crossval`; Supplement S6 | No cluster input in training; absent subgroups not scored as zero |
| R1/R3 comparison replies | All public/vehicle lightweight baselines on the same protocol/platform | documented interfaces and artifacts only | NOT IMPLEMENTED as the named external methods; original complete architectures/code and original measurements required |
| Whole response | Public code and detailed formulas | English release + Times New Roman Supplementary Methods PDF and editable Markdown | Repository: https://github.com/Khalil0-afk/DA-LSS |

## Source inconsistencies requiring author reconciliation

1. R1.4 pp.22-23 introduces vehicle-domain adaptation, but its retained manuscript excerpt and several other replies still describe no vehicle adaptation. Both executable protocols are retained and named separately. The new adaptation procedure follows the substantive reply and Table R; it must not be reported as completely unseen-domain generalization.
2. The response's vehicle-specification table lists six C11 vehicles, while manuscript passages list six A05 vehicles. The release does not invent which cohort is correct. A real release needs the authorized vehicle identity manifest.
3. The response discusses S4/Mamba, whereas the current manuscript comparison narrative names FESD/MINE-Graph-Mamba and its configuration table still names S4/Mamba. No simplified network is relabeled as any of these complete external methods.
4. Some pasted older preprocessing formulas use a forward index despite the clarified trailing-average prose. The implemented direction follows the clarified rule and current manuscript.
5. The supplied documents do not uniquely specify the original gate statistics, original complete architecture or original tuning space/budget. Declared completion choices are not original recovered settings.

## Empirical assets still needed

Original data inclusion/physical-cell identity lists, original study splits and seeds, actual four/20-vehicle identities and records, authorized CAN specification, native HUST layout, full external baseline implementations, original complete network/checkpoint and Jetson measurement configuration remain author inputs. Synthetic validation demonstrates executable mechanisms and isolation rules, not the reported error ranking or hardware table.
