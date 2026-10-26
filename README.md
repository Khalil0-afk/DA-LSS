# DA-LSS

English-only open-source implementation aligned with the author-provided revision response. Version 0.3.0.

This release includes the DA-LSS model, public-data cross-validation, pre-deployment vehicle adaptation, identity-isolated online replay, CAN engineering-unit conversion, timestamp/reference matching, capacity definitions, resource measurement and supplementary methods. Python source files contain no comments or docstrings. The primary [Supplementary Methods PDF](supplementary/Supplementary_Methods.pdf) uses embedded Times New Roman for body text and equation letters, with Times-compatible mathematical symbols. All32 equations are vector text. Other explanatory documents, license text and archived logs remain Markdown; the [editable supplement source](supplementary/SUPPLEMENTARY_METHODS.md) is retained.

The implementation is independently reconstructed from the supplied documents. This repository includes supplied processed MIT and HUST feature tables. Verification uses explicitly synthetic records; the original raw archives, private vehicle measurements, original complete architecture and external baseline implementations have not been independently validated here. The release does not certify the manuscript's reported accuracy, parameter count or Jetson measurements. See the [commitment-to-artifact tracker](docs/response_commitments.md) for implemented items and remaining author assets.

## Datasets

```text
data/
  MIT data/
  HUST data/
  Leapmotor_EV/
```

The [dataset guide](data/README.md) documents the included125 MIT CSVs and77 HUST CSVs, their processed feature format, and the distinction from the canonical training input. The real-world vehicle BMS/CAN dataset is available from the corresponding author upon reasonable request. Original provider terms apply to the public datasets; the code's MIT license does not relicense third-party data.

## Installation

Python 3.9-3.11 is recommended. Install the CPU or CUDA build of PyTorch appropriate for your machine, then install the project and optional analysis tools:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[all]"
python -m unittest discover -s tests -v
```

The core requires NumPy/PyTorch. Optional extras are `analysis`, `data`, `tuning` and `profiling`. The `all` extra covers experiments and profiling. The optional `supplementary` extra is only needed to rebuild the PDF; training and inference do not require it. Rebuilding requires installed Times New Roman font files, which are not distributed with the project.

## Reproduce the software checks

```bash
python scripts/reproduce.py --output runs/check --extended
python scripts/check_response_workflows.py --output runs/response-check
```

Use a fresh output directory for each run. The first command runs the original core workflow, including synthetic training, 40 partition combinations, ablations, sensitivity, source-validation tuning, export, online replay and reports. The second checks response-specific CAN/capacity/power/resource paths and the exact four-vehicle, 128-window, 40-update adaptation budget. Neither downloads private data nor uploads artifacts. Fast checks are software validation, not evidence of method superiority.

## Public-data experiments

```bash
python -m da_lss prepare --input data/segments.csv --config configs/preprocessing.json --output data/features.csv
python -m da_lss train --features data/features.csv --source MIT --target HUST --config configs/paper.json --output runs/train --device cuda
python -m da_lss crossval --features data/features.csv --source MIT --target HUST --config configs/paper.json --both-directions --output runs/crossval --device cuda
```

The source has five battery folds and the target four. Each direction crosses five source configurations with four target configurations. Source training fits normalization; source training and unlabeled adaptation fit gate statistics; source validation alone selects checkpoints and hyperparameters. All windows of one physical battery keep the same role within a combination. Prediction files are saved before test scoring. Manifests contain battery identities, per-battery window counts, actual adaptation fractions, seeds, strides, configurations and hashes. More commands are in [Experiments](docs/experiments.md).

## Vehicle protocols

The supplied documents contain both direct frozen evaluation and a newer reply specifying pre-deployment unsupervised adaptation. Keep them separate. The response-specific adaptation command is:

```bash
python -m da_lss adapt-vehicle --checkpoint runs/train/best.pt --source-features data/features.csv --adaptation-features data/four_adaptation_vehicles.csv --test-vehicles data/reserved_20_vehicle_ids.json --config configs/vehicle_adaptation.json --output runs/vehicle-adaptation --device cuda
python -m da_lss replay --checkpoint runs/vehicle-adaptation/best.pt --input data/held_out_ev_features.csv --output runs/held-out-replay --device cuda
```

The adaptation input requires a `vehicle_id` on every row. Four adaptation vehicles provide exactly 32 selected stride-32 windows each. A list of 20 reserved test identities is checked for overlap without loading test measurements. Initialization and source train/validation identities come from the pretrained checkpoint and its source manifest. Adam uses 1e-4 learning rate and weight decay, ten epochs of four updates. EV labels are discarded; source validation selects the adapted checkpoint. The source normalization remains frozen, and selected-window/gate provenance is recorded.

Online replay keeps independent buffers for each **vehicle and cell**, including when different vehicles reuse the same cell number. It predicts first at 128 valid vectors and subsequently every 32 vectors, with batch size one and no padding. Model/gate state stays frozen; the discriminator is inactive. Results include vehicle, charging-protocol and vehicle-by-protocol metrics when references and metadata are provided.

## Acquisition, reference definitions and resources

```bash
python -m da_lss decode-can --input data/can.csv --mapping data/authorized_mapping.json --output data/signals.csv
python -m da_lss match-signals --input data/signals.csv --max-age-s 1 --output data/matched.csv
python -m da_lss matched-segments --input data/matched.csv --output data/segments.csv
python -m da_lss capacity-soh --input data/capacity_segments.csv --output runs/capacity.json
python -m da_lss match-references --input runs/held-out-replay/predictions.csv --references data/bms_references.csv --max-age-s 60 --output runs/reference-matched.csv
python -m da_lss resources --checkpoint runs/train/best.pt --raw-segments-npz data/window_segments.npz --exported-model runs/export/model.ts --output runs/resources
python -m da_lss power-energy --input data/measured_power.csv --completed-windows 100 --output runs/energy.json
```

The time tolerances above are examples and must follow the acquisition protocol. Byte-aligned CAN maps are user supplied; `configs/can_mapping.synthetic.json` is only a synthetic fixture. Cell SOH is not populated from pack SOH. Charge/SOC integration estimates and BMS-reported evaluation references remain distinct. Resource output states operation-count coverage, timing boundaries and memory interfaces. Unmeasured power is null; power integration requires a measured post-warmup trace and the actual complete-window count. See [Vehicle data](docs/vehicle_data.md) and Supplementary Sections S1-S2, S7-S9.

## Artifacts and scope

- [Methods and completion choices](docs/method.md)
- [Data schema](docs/data.md)
- [Response commitments and unresolved inconsistencies](docs/response_commitments.md)
- [Verification record](docs/verification.md)

Repository: [Khalil0-afk/DA-LSS](https://github.com/Khalil0-afk/DA-LSS). The code uses the [MIT License](LICENSE.md); that license does not cover the manuscript, review correspondence, third-party data or third-party models. No original source document or private vehicle record is distributed. The repository includes English documentation, the supplementary PDF and supplied public-data feature tables. Software checks generate synthetic inputs and models in the requested output directory at runtime.
