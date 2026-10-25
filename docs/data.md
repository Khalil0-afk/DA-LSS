# Data contract

The repository includes125 processed MIT feature CSVs and77 processed HUST feature CSVs, documented in [data/README.md](../data/README.md). These17-column tables require explicit physical-cell/cycle identities and a documented capacity-to-SOH conversion before entering the canonical training interface below. Raw within-cycle archives and private EV records are not included.

## Canonical raw segments

CSV header:

```csv
domain,battery_id,cycle,time_s,voltage_v,current_a,soh
MIT,cell_001,1,0,3.20,1.50,0.998
MIT,cell_001,1,1,3.21,1.50,0.998
```

The two rows are schema examples, not a sufficient segment. Default smoothing requires at least 15 retained observations per segment. A `(domain,battery_id,cycle)` group must represent exactly one consistently selected, completed charging segment. Time is seconds, voltage is **cell** voltage in volts, current is amperes; positive means charging. The caller must exclude rest/discharge and define segment boundaries consistently. This project does not infer device-specific charging protocols.

Rows are sorted by time and timestamps must be finite/unique. Voltage/current outlier masks are combined so alignment is preserved. Statistics use population standard deviation. After rejection, a valid moving average of width 15 yields N-14 points; no zero padding. Equation (2) and pseudocode disagree on forward/backward indices: valid trailing averaging is the declared choice. Entropy uses 32 fixed bins in [2.0,4.5] V and natural logarithms, with no fitting to target data. Out-of-range smoothed voltage raises an error. Customize this fixed range with `--voltage-range` for other chemistries.

The SOH column is optional; if present, values within a segment must agree. Select a capacity denominator based on the actual dataset and document it. Do not silently assume nominal, first-cycle and maximum capacity are interchangeable. Do not use target SOH to choose segments, normalize features or filter test batteries.

## Canonical features

```csv
domain,battery_id,cycle,mean_voltage,mean_current,voltage_entropy,voltage_std,soh
MIT,cell_001,1,3.30,1.50,1.20,0.04,0.998
HUST,cell_002,1,3.32,1.47,1.15,0.03,
```

One row per completed segment/cycle; four finite features in exactly this order. Battery IDs are unique within a domain. Cycles must be finite and unique, then are sorted numerically. Windows never cross batteries. For each battery, labels must be either entirely missing or complete positive SOH fractions (values over 1.5 rejected as likely scaling errors). Source train/validation require labels. Target adaptation works with empty SOH. Target test labels are optional for prediction, but complete labels are required to summarize cross-validation.

The model does not use battery ID, cycle index, SOH or retrospective group as an input feature. It uses only the four columns. Batteries shorter than the configured window are recorded as excluded before splitting; missing cycles may occur within windows, but TMR only compares endpoints separated by exactly one cycle. Large chronological gaps should be addressed explicitly in the user's data protocol.

The raw conversion is a canonical-format importer. A separate `import-mit` adapter follows the dataset authors' MATLAB v7.3 HDF5 layout and is tested against a matching structural fixture, not a real original archive. Native HUST layout remains unverified and uses the canonical interface. No real data download or redistribution is bundled. Vehicle identities, CAN conversion and reference matching are covered in `vehicle_data.md`.

## Primary dataset sources

These are sources for obtaining and documenting data, not evidence that this project's experiments have run on them:

- MIT/Stanford study: [Severson et al., Nature Energy (2019)](https://www.nature.com/articles/s41560-019-0356-8). The study describes 124 cells. Its [authors' repository](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation) links data and preprocessing resources.
- HUST study: [Ma et al., Energy & Environmental Science (2022)](https://pubs.rsc.org/en/content/articlehtml/2022/ee/d2ee01676a). The study describes 77 cells. The associated [Mendeley dataset V2](https://data.mendeley.com/datasets/nsc7hnsg4s/2) has DOI 10.17632/nsc7hnsg4s.2.

The supplied manuscript's MIT grouping figure sums to 125 cells, whereas the original MIT study describes 124. This repository does not silently reconcile that discrepancy; the real inclusion list is needed. Data terms belong to the respective providers. Private EV records are not provided here.

For EV evaluation, keep the model and scaler/gate statistics frozen. BMS SOH is an estimated reference, not an independently measured capacity ground truth. Cell identity and prediction timestamp matching must be established before scoring.
