# Polynomial Regression — BT2024108

Complete submission for the Polynomial Regression assignment.

## Methodology

For each personalised dataset, polynomial degrees allowed by the assignment were evaluated with three approaches:

1. Polynomial + Linear Regression
2. Polynomial + Ridge Regression
3. Polynomial + Lasso Regression

Five-fold shuffled cross-validation (`random_state=42`) was used. Ridge/Lasso alpha values were `{0.01, 0.1, 1, 10}`. Standardisation is inside each sklearn Pipeline to avoid validation leakage. The lowest mean CV MSE determines the final model; CV R² is a supporting metric.

## Final models

| Dataset | Method | Degree | Alpha | CV MSE | CV R² |
|---|---|---:|---:|---:|---:|
| var1 | Lasso | 5 | 0.01 | 0.321413 | 0.965634 |
| var2 | Ridge | 11 | 1 | 0.238684 | 0.994651 |

## Reproduce

```bash
pip install -r requirements.txt
python src/evaluate.py
python src/train.py
python src/predict.py
```

## Repository

https://github.com/car-ui/ML-Assignment
