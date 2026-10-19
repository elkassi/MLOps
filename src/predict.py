from pathlib import Path

import joblib
from sklearn.datasets import load_iris

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.pkl"


def main():
    model = joblib.load(MODEL_PATH)
    iris = load_iris()
    sample = iris.data[0:1]
    prediction = int(model.predict(sample)[0])
    print(f"Classe : {prediction}")
    print(f"Nom : {iris.target_names[prediction]}")


if __name__ == "__main__":
    main()
