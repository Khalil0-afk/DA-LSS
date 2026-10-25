# Vehicle records and acquisition workflow

All timestamps are numeric seconds on one documented clock. Record units are cell voltage in V, pack main current in A and SOH/SOC as fractions. The caller supplies charging-segment identities and verifies the series-current assumptions in Supplement S1. No OEM CAN identifiers or calibrations were supplied.

## Byte-aligned CAN CSV and map

```csv
vehicle_id,battery_id,segment_id,timestamp,frame_id,data_hex
synthetic_vehicle,cell_01,1,0,0x100,e40c9600
```

The example bytes and frame identifier are synthetic. The JSON map declares frame ID, signal name/scope, byte offset/length, byte order, signedness, scale and offset. Decoding applies `engineering = raw*scale+offset`. Only byte-aligned fields are supported; this is not a general Motorola/DBC bit-field parser. Use the actual authorized acquisition specification rather than guessing a map from a manuscript.

Decoded observations have `vehicle_id,battery_id,segment_id,timestamp,signal,scope,value`. Scope is `cell` or `pack`; names used by matching are `cell_voltage_v` and `pack_current_a`. Offline records are ordered by event timestamp, same-time observations are registered together, and a cell voltage uses only same-vehicle pack current at or before its timestamp within the declared maximum age. Missing/stale matches are counted. `matched-segments` converts declared numeric segment IDs into the canonical raw format and leaves SOH empty; it does not infer charging boundaries or references.

## Capacity CSV

```csv
vehicle_id,battery_id,segment_id,scope,timestamp,current_a,soc,nominal_capacity_ah
synthetic_vehicle,cell_01,1,cell,0,1,0.2,2
synthetic_vehicle,cell_01,1,cell,1800,1,0.45,2
synthetic_vehicle,cell_01,1,cell,3600,1,0.7,2
```

For this illustrative slow sampling, explicitly set `--max-gap-s 1800`; the normal default60 would correctly reject it. `scope` distinguishes pack and cell definitions. Endpoint SOC must refer to the same object. Default minimum absolute SOC excursion is0.05. The trapezoidal current integral is divided by signed endpoint SOC change, then by that object's nominal capacity. This returns an integration-derived estimate, not the independently supplied BMS reference. Never assign a pack estimate to all cells.

## Feature and vehicle identity metadata

Vehicle adaptation uses the canonical feature header plus `vehicle_id`. Each `(domain,battery_id)` must identify one vehicle's physical cell uniquely across the adaptation table. Domain-wide source manifests retain source training/validation identity assignments. The supplied reserved-test JSON is a list of exactly20 unique vehicle IDs; the default adaptation input must contain exactly4 different, nonoverlapping vehicles. Each provides at least32 distinct windows at stride32. If one cell supplies all windows, at least1120 valid chronological vectors are needed. Insufficient data is an error; duplicate/replacement selections do not inflate the registered window pool.

The registered pool has128 distinct windows. Subsequent minibatch draws use vehicle-uniform replacement sampling. Ten epochs with four updates each are completed, independently of whether each registered window has been drawn. Source-only validation selects the adapted checkpoint. The default source/target minibatch is128+128, an explicit completion choice absent from the response's adaptation table. Source TMR is retained; unregistered target successor windows are not introduced.

Online replay requires `domain,battery_id,vehicle_id,timestamp` and the four features. Optional `soh` and `charging_protocol` are post-prediction evaluation metadata. Cell names reused across vehicles remain separate FIFO streams. Suggested protocol names are `AC_slow`, `DC_fast` and `DC_high_power`; names are metadata, not model inputs. A window can contain mixed protocol history, and grouping is assigned by its endpoint record.

## BMS references

```csv
vehicle_id,battery_id,timestamp,bms_soh,scope
synthetic_vehicle,cell_01,128,0.95,cell
```

`match-references` accepts already saved predictions and references. It uses only same-vehicle/same-cell past-or-equal reference timestamps within user-specified tolerance. Unknown/too-old references remain empty. Pack-scoped rows and duplicate cell reference timestamps are rejected. This function cannot affect the already generated prediction file or model state. BMS-reference agreement is not measured-capacity validation.

## Resource and power inputs

`resources --raw-segments-npz` accepts arrays named `voltage` and `current`, each with one row per completed sequence segment, for example shape(128,32). Loading uses `allow_pickle=False`. This permits inclusion of raw-segment cleaning and feature construction in the timed path. Without these arrays, timing starts at precomputed features and the report states that scope. Acquisition, CAN decoding and asynchronous matching are outside this timing harness.

Measured power uses `timestamp_s,power_w`. Pass the number of fully completed windows in the same post-warmup interval. Samples must be finite, nonnegative and strictly chronological. Integration uses actual intervals and does not subtract idle power. The synthetic power fixture tests the arithmetic only and is never reported as a hardware measurement.
