# MLOps TP1 - Iris

This project trains a reproducible Random Forest classifier on the Iris dataset, records experiments in MLflow, saves a model, makes a prediction, and inspects reference data statistics for drift.

## Setup on Windows

```powershell
cd C:\Users\DELL\Desktop\mlops-tp1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The installed Python version for this project is 3.14.4. Use the virtual environment's Python executable in the commands below; PowerShell activation is optional.

## Train and compare experiments

```powershell
.\.venv\Scripts\python.exe src\train.py
```

The script evaluates 50, 100, and 200 trees on the same stratified 80/20 split with `random_state=42`. It logs each run to the local SQLite MLflow database, writes `experiments/run_comparison.csv`, and saves the best model to `models/iris_model.pkl`.

| Trees | Accuracy |
| ---: | ---: |
| 50 | 0.90 |
| 100 | 0.90 |
| 200 | 0.90 |

## View MLflow

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///experiments/mlflow.db --host 127.0.0.1 --port 5001
```

Open <http://127.0.0.1:5001> and select the `Iris_MLOps` experiment. Port 5001 avoids the Flask TP1 API on port 5000. You should see three successful runs with `n_estimators` and `accuracy` values.

## Predict and inspect drift inputs

```powershell
.\.venv\Scripts\python.exe src\predict.py
.\.venv\Scripts\python.exe src\drift.py
```

`predict.py` loads the saved model and predicts the first Iris sample as setosa. `drift.py` prints summary statistics for the four input measurements. The interpretation is in `experiments/drift_notes.md`.

## Files for submission

Submit the source files, `requirements.txt`, `.gitignore`, the generated `models/iris_model.pkl`, the comparison CSV, and MLflow screenshots. The model and tracking database are ignored by Git as requested in the handout, so keep the local project folder for the full submission.
