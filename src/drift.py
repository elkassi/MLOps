import pandas as pd
from sklearn.datasets import load_iris


def main():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    print(data.describe().round(2).to_string())


if __name__ == "__main__":
    main()
