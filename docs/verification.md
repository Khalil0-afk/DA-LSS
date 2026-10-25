# Verification record

Release:0.3.0, Times New Roman supplementary-PDF edition. Date:2026-10-25. All experiment inputs used for verification are synthetic. No manuscript result is substituted for a measured output. English text and comment-free Python source are checked during release packaging. This typography revision preserves the model and experiment implementation.

## Environment

Windows; Python3.9.23; PyTorch2.8.0+cu128; NumPy1.24.4; Matplotlib3.7.5; scikit-learn1.3.2; h5py3.11.0; Optuna4.5.0. Core CPU checks use four Torch threads. The supplement uses embedded Times New Roman body/equation-letter fonts with Times-compatible operator symbols. All32 equations are vector text; their editable Markdown source is preserved.

## Completed checks

| Check | Result | Scope |
| --- | --- | --- |
| Unit/integration suite | 40 tests passed | Numerical ZOH, GRL, TMR, split isolation, frozen state, export, recovery, CAN, capacity, power, vehicle adaptation |
| Core reproduction script | 17 stages passed | Includes40 synthetic partition runs, ablation, sensitivity, source-validation tuning and post-hoc reports |
| Response-specific script | 8 stages passed | CAN decode, past-only matching, segment/features, capacity, power arithmetic, resources and vehicle adaptation |
| Default vehicle adaptation budget | 4 vehicles,32 windows each,128 total,40 updates | Actual default10 epochs times4 updates, source-validation checkpoint selection |
| Adaptation label perturbation | Identical adapted model state | Target SOH changed without affecting any training parameter/buffer |
| Vehicle-level separation | Overlap rejected | Test identities checked without loading reserved EV measurements |
| Reused cell identifiers | Separate per-vehicle histories | Same cell number on two vehicles does not combine FIFO data |
| BMS-reference matching | Past/equal, same vehicle/cell only | Future matches and pack-scoped broadcast references rejected |
| Partial-SOC capacity | Charge and discharge equations passed | Sign, excursion and gap checks included |
| Power integration | Nonuniform trapezoidal integration passed |55J test trace gives11,000mJ per window for5 windows; synthetic arithmetic fixture only |
| Resource harness | Executed on local CPU | Includes raw-segment feature extraction when supplied; null unmeasured watts |
| Model wheel | Build succeeded | da_lss-0.3.0, no dependency downloads during the build |
| Supplement | Times New Roman PDF and Markdown source |32 vector equations; embedded fonts and English-only content |

Machine-readable evidence and per-stage logs are in `docs/evidence/`. They explicitly mark synthetic data and the distinction between arithmetic power fixtures and actual telemetry. The full development run directories are excluded from the source release.

For the Markdown documentation edition, the40-test suite and eight-stage core run were executed again successfully, and the wheel was rebuilt with `LICENSE.md`. The new runtime logs are fenced Markdown in `docs/evidence/markdown-logs/`. All32 supplementary equations match the preceding release's editable Markdown byte-for-byte. Local documentation links, English-only text and comment-free Python sources were checked before packaging. Executable configuration/data formats and the pip requirements list retain their required machine-readable formats.

The subsequent typography revision changes only the supplement renderer, document references and optional build dependencies. The PDF is rendered page by page for visual inspection. Font metadata is recorded in `supplementary/font_audit.json`; all equation letters use Times New Roman. Two membership symbols and one sized radical use matching STIX operator glyphs. No equation screenshots or raster formula images are used.

## Resource interpretation

For the current64-hidden-state DA-LSS architecture, the exercised inference path has18,178 parameters excluding the domain classifier. The operation report counts2,261,568 MACs per128-vector window and approximately4.640 accounted MFLOPs under its disclosed convention. Normalization and transcendental implementation costs remain listed as unaccounted; this is not the response's full24.6832 MFLOPs value.

The tested local run produced forward latency around2.6ms and raw-segment-start pipeline latency around13.8ms, with process RSS around435MB and an exported example around0.103MB. Exact values depend on the measured process and are stored in `local_resources.json`. They include the stated runtime/process scope and exclude CAN acquisition and asynchronous matching. They are not Jetson results and must not be presented as a reproduction of the response's latency/memory/power table.

## Important changes verified in this release

The previous replay key used domain/cell identity only. That was insufficient when different vehicles reused cell numbers. Replay now keys its FIFO by both vehicle and cell and preserves those identifiers in prediction/reference records. This is covered by a regression test. The public-data battery identity rules remain unchanged.

The response-specific adaptation workflow continues from a source-selected pretrained checkpoint rather than reinitializing training. It freezes the original source scaler, selects distinct adaptation windows before updates, discards EV labels and completes the fixed update budget. The output is separate from the direct frozen-generalization protocol because the supplied documents contain both descriptions.

## Remaining empirical scope

The real four/20-vehicle identity list and records, authorized OEM CAN map, original complete model, original native-data corrections and all named external baseline implementations were not supplied. The claimed original error values, complete parameter/operation count and Jetson measurements have not been independently regenerated. These needs are explicitly mapped in `response_commitments.md`; synthetic code validation does not satisfy them by itself.
