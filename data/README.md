# Datasets

| Directory | Included content |
| --- | --- |
| [MIT data](MIT%20data/README.md) |125 supplied processed feature CSVs across three experiment batches |
| [HUST data](HUST%20data/README.md) |77 supplied processed feature CSVs |
| [Leapmotor_EV](Leapmotor_EV/README.md) |Access statement; private vehicle measurements are not included |

The real-world vehicle BMS/CAN dataset is available from the corresponding author upon reasonable request.

## Processed feature format

Each supplied public-data CSV has17 columns, including voltage/current summary statistics and a `capacity` column. These files are processed feature tables rather than raw voltage/current time series. The four model features correspond to `voltage mean`, `current mean`, `voltage entropy` and `voltage std`.

The supplied CSVs do not contain explicit `domain`, physical `battery_id`, cycle-index or `soh` fields. Before using them with the canonical training interface, establish the physical battery identity and chronological cycle mapping, preserve the original row order, specify the capacity denominator from the data protocol, and construct the schema in [docs/data.md](../docs/data.md). Do not silently equate raw capacity with normalized SOH or assume that row numbers always equal original cycle numbers.

The provider's original terms govern these third-party datasets. The repository's MIT license covers the new implementation and does not relicense the data. Software checks generate synthetic inputs in their output directories at runtime.
