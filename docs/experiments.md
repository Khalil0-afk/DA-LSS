# Experiment commands and audit artifacts

Run from the project root. Every new experiment requires a new output directory. `configs/paper.json` uses the manuscript's stated common settings together with explicitly documented completion choices. `configs/smoke.json` is strictly for software checks.

## Training and recovery

```bash
python -m da_lss train --features data/features.csv --source MIT --target HUST --config configs/paper.json --source-fold 0 --target-fold 0 --output runs/one --device cuda
python -m da_lss train --features data/features.csv --source MIT --target HUST --config configs/paper.json --source-fold 0 --target-fold 0 --output runs/one --device cuda --resume
```

Recovery requires identical data, split, configuration and device. `best.pt` contains the source-validation-selected model/scaler/gate; `last.pt` additionally stores optimizer and random states for the latest complete epoch. An interrupted partial epoch restarts from the last complete checkpoint. The integration test compares recovered and uninterrupted model/history states exactly on the tested CPU environment.

## Cross-validation, ablation and sensitivity

```bash
python -m da_lss crossval --features data/features.csv --source MIT --target HUST --config configs/paper.json --both-directions --output runs/crossval --device cuda
python -m da_lss suite --suite ablation --features data/features.csv --source MIT --target HUST --config configs/paper.json --all-folds --both-directions --output runs/ablation --device cuda
python -m da_lss suite --suite sensitivity --features data/features.csv --source MIT --target HUST --config configs/paper.json --output runs/sensitivity --device cuda
python -m da_lss sensitivity-report --suite runs/sensitivity --features data/features.csv --output runs/sensitivity-figures
```

The ablation suite includes proposed, no adversarial alignment, no TMR, LSTM backbone replacement, source-only and no health gate. Removing alignment retains unlabeled gate/TMR use and therefore differs from source-only. LSTM replacement removes unused SSM parameters and auxiliary-health loss. LSTM/Bi-GRU/Transformer are backbone controls, not relabeled external complete methods.

The sensitivity suite covers the five settings in Supplement S5. A display battery is chosen with a fixed seed from the specified target-test fold before any prediction. In a short smoke run TMR can be zero and all settings can agree; such runs do not establish scientific effectiveness.

## Source-validation-only tuning

```bash
python -m da_lss tune --features data/features.csv --source MIT --target HUST --config configs/paper.json --trials 10 --output runs/tuning --device cuda
python -m da_lss train --features data/features.csv --source MIT --target HUST --config runs/tuning/best_config.json --output runs/tuned-final --device cuda
```

TPE search settings are disclosed in `configs/tuning_protocol.json` and each run's `search_protocol.json`; the actual trial budget, seed and stopping patience are stored. Trials generate no test prediction files. Search ranges and budget are declared reconstructions because the response promises disclosure but does not itself provide the original numeric search specification. Tune independently within each outer split to preserve its validation isolation.

## Retrospective evaluation and plots

Only after predictions are finalized:

```bash
python -m da_lss cluster --features data/features.csv --output runs/groups.json
python -m da_lss group-crossval --runs runs/crossval --features data/features.csv --grouping runs/groups.json --output runs/group-summary
python -m da_lss report --run runs/one --features data/features.csv --output runs/figures
python -m da_lss heatmap --checkpoint runs/one/best.pt --features data/features.csv --battery HUST/cell_001 --output runs/heatmap
```

The group summary contains both pooled-window and battery-macro RMSE, sample SD and explicit RMSE-times100 units. A subgroup absent from a partition contributes no value rather than a false zero; contributing-partition counts are saved. Repeated partitions overlap and their SD is not an independent-replicate confidence interval. Correlation heatmaps describe internal representations and do not prove causal long-range dependence.

## Deployment

```bash
python -m da_lss export --checkpoint runs/one/best.pt --output runs/export
python -m da_lss predict --checkpoint runs/one/best.pt --features data/unseen_features.csv --output runs/predictions
python -m da_lss replay --checkpoint runs/one/best.pt --input data/ev_features.csv --output runs/replay
python -m da_lss resources --checkpoint runs/one/best.pt --exported-model runs/export/model.ts --output runs/resources --device cpu
```

Export embeds normalization and removes the discriminator. Traced sequence length is fixed and stated in metadata; batch1/2 reload parity is tested. Resource results explicitly distinguish forward timing, feature-start timing and raw-segment-start timing. Sampled process RSS, CUDA allocator memory, platform power and exported bytes have separate scopes. No unmeasured hardware value is inferred from a manuscript table.

Vehicle adaptation and acquisition commands are documented separately in the README and `docs/vehicle_data.md`. The default adaptation budget follows response R1.4 Table R exactly; direct frozen replay remains a separate protocol. Both require real held-out data to establish empirical performance.
