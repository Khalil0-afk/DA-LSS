# Demo

````text
{"epoch": 1, "task": 0.0009027037061362837, "domain": 0.6994386613368988, "tmr": 1.0647636372596025e-06, "auxiliary": 0.002584706603859862, "source_val_rmse": 0.044413940532082176, "supported_health_points": 4}
{"epoch": 2, "task": 0.0005261186112572129, "domain": 0.7064668039480845, "tmr": 1.3325839972821996e-05, "auxiliary": 0.0011870826225883018, "source_val_rmse": 0.08286127598430644, "supported_health_points": 3}
{"epoch": 3, "task": 0.000323143972006316, "domain": 0.7048482000827789, "tmr": 0.0, "auxiliary": 0.0002669527457328513, "source_val_rmse": 0.0747852717502451, "supported_health_points": 2}
{
  "elapsed_seconds": 0.9195662999999996,
  "best_source_val_rmse": 0.044413940532082176,
  "trainable_parameters": 22403,
  "inference_parameters": 18178,
  "source_test": {
    "labeled_windows": 46,
    "total_windows": 46,
    "pooled_windows": {
      "rmse": 0.022547646029588354,
      "mae": 0.021453739508338596,
      "mape_percent": 2.5981259638006926
    },
    "per_battery": {
      "synthetic_source/cell_01": {
        "rmse": 0.02420614050127383,
        "mae": 0.02353606016739555,
        "mape_percent": 2.88059218483056
      },
      "synthetic_source/cell_09": {
        "rmse": 0.02075705771499934,
        "mae": 0.019371418849281643,
        "mape_percent": 2.315659742770826
      }
    },
    "macro_battery": {
      "rmse": 0.022481599108136584,
      "mae": 0.021453739508338596,
      "mape_percent": 2.598125963800693
    }
  },
  "target_test": {
    "labeled_windows": 46,
    "total_windows": 46,
    "pooled_windows": {
      "rmse": 0.09337715080352155,
      "mae": 0.08974241951237553,
      "mape_percent": 10.59274573763432
    },
    "per_battery": {
      "synthetic_target/cell_00": {
        "rmse": 0.06539196131286477,
        "mae": 0.0647851798845374,
        "mape_percent": 8.005502406698495
      },
      "synthetic_target/cell_01": {
        "rmse": 0.11472783437345972,
        "mae": 0.11469965914021367,
        "mape_percent": 13.17998906857014
      }
    },
    "macro_battery": {
      "rmse": 0.09005989784316225,
      "mae": 0.08974241951237553,
      "mape_percent": 10.592745737634317
    }
  }
}

````
