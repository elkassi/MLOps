from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.pkl"
TRACKING_DB = PROJECT_ROOT / "experiments" / "mlflow.db"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "run_comparison.csv"
TREE_COUNTS = (50, 100, 200)


def main():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    mlflow.set_experiment("Iris_MLOps")

    results = []
    best_model = None
    best_accuracy = -1.0
    for n_estimators in TREE_COUNTS:
        with mlflow.start_run(run_name=f"RandomForest_{n_estimators}") as run:
            model = RandomForestClassifier(
                n_estimators=n_estimators, random_state=42
            )
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(
                model, name="model", pip_requirements=["scikit-learn"],
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

            results.append({
                "run_id": run.info.run_id,
                "n_estimators": n_estimators,
                "accuracy": accuracy,
            })
            print(f"{n_estimators} trees: accuracy = {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_model = model
                best_accuracy = accuracy
                best_predictions = predictions

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print("\nBest model classification report:")
    print(classification_report(
        y_test, best_predictions, target_names=iris.target_names
    ))
    print(f"Model saved to {MODEL_PATH}")
    print(f"Run comparison saved to {RESULTS_PATH}")
    print(f"MLflow tracking database: {TRACKING_DB}")


if __name__ == "__main__":
    main()
