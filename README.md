# CatBoost 78-Hour Voltage Forecasting for Soil MFC

This repository contains a production-oriented CatBoost regression pipeline for forecasting **SMFC voltage 78 hours ahead** (13 timesteps at 6-hour frequency) from historical voltage measurements.

## Dataset

Input file: `data_smfc_data.csv`

Schema:
- `TIME`: custom timestamp text in format `day-X; H(am|pm)`
- `VOLTAGE`: voltage reading (float)

## Data inspection findings

From the supplied dataset:
- Total rows: **101** observations (excluding header)
- Time cadence: predominantly **6-hour intervals** (`1am`, `7am`, `1pm`, `7pm`)
- Voltage range: approximately **0.046 to 0.961**
- Expected columns present: `TIME`, `VOLTAGE`

The training script validates:
- Required columns
- Timestamp parsing integrity
- Numeric voltage coercion
- Duplicate timestamps
- Missing values and irregular sampling intervals

## What the pipeline does

The script `train_catboost_smfc.py` performs end-to-end modeling:

1. **Load + validate data**
2. **Engineer time-series features**
   - Lagged voltages: 1, 2, 3, 6, 12, 13, 24 steps
   - Rolling windows (3/6/12/24): mean/std/min/max
   - Dynamics: first difference, second difference, percent rate of change, acceleration
   - MFC stability signals:
     - rolling range
     - rolling coefficient of variation
     - inverse rolling volatility (`voltage_stability_12`)
     - EMA crossover gap (`ema_6 - ema_12`)
   - Time cyclical features from parsed timestamps:
     - hour sin/cos
     - day-of-week sin/cos
3. **Create target**
   - `target_78h = voltage.shift(-13)`
4. **Chronological split** (no leakage)
   - Train: 70%
   - Validation: 15%
   - Test: 15%
5. **Hyperparameter tuning**
   - Searches several CatBoost configurations across depth / learning rate / regularization / iterations
   - Selects best model by **validation RMSE**
6. **Evaluation**
   - RMSE, MAE, R², MAPE on validation and test
   - Prediction and residual plots
7. **Artifact export** for production inference

## Installation (local, Jupyter, or Colab)

```bash
pip install pandas numpy scikit-learn catboost matplotlib
```

## Training

```bash
python train_catboost_smfc.py --data data_smfc_data.csv --output artifacts
```

Generated artifacts:
- `artifacts/catboost_smfc_78h.cbm` (saved model)
- `artifacts/feature_columns.json`
- `artifacts/metrics_summary.json`
- `artifacts/test_predictions.csv`
- `artifacts/feature_importance.csv`
- `artifacts/actual_vs_pred_validation.png`
- `artifacts/actual_vs_pred_test.png`
- `artifacts/residuals_validation.png`
- `artifacts/residuals_test.png`

## Inference on new data

Use `predict_from_csv` from `train_catboost_smfc.py`.

```python
from pathlib import Path
from train_catboost_smfc import predict_from_csv

prediction = predict_from_csv(
    model_path=Path("artifacts/catboost_smfc_78h.cbm"),
    feature_cols_path=Path("artifacts/feature_columns.json"),
    input_csv=Path("new_voltage_history.csv"),
)
print("Predicted voltage 78h ahead:", prediction)
```

### New data requirements

- Must include columns `TIME` and `VOLTAGE`
- `TIME` format must match: `day-X; H(am|pm)`
- Must contain enough historical points for lag/rolling features (at least ~24+ rows recommended)

## Modeling assumptions and limitations

- Data cadence assumed to be 6-hour measurements.
- Dataset is relatively small, so uncertainty can be high during regime shifts.
- The model is univariate (voltage-only). Exogenous drivers (temperature, substrate load, pH, etc.) are not included.
- Sudden operating mode changes can degrade forecast quality.
- If zeros/near-zeros occur in voltage, MAPE may become unstable; implementation guards denominator with epsilon.

## Failure modes and safeguards

The script includes checks and explicit errors for:
- Missing files
- Missing/invalid columns
- Unparseable time strings
- Insufficient rows after feature generation
- Missing inference features

## Notes for production

- Retrain periodically as new MFC operating data arrives.
- Monitor drift with rolling residual statistics.
- Version model artifacts and feature schema together.
