import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Feature import Feature
from EM import EM


def get_columns(name: str):
    return [
        f"d_{name}",
        f"d_{name}_partner",
        f"d_{name}_o",
        f"d_{name}_important",
        f"d_pref_o_{name}",
    ]


def balance_dataset(X: np.ndarray, Y: np.ndarray, random_state: int = 42):
    np.random.seed(random_state)
    pos_idx = np.where(Y == 1)[0]
    neg_idx = np.where(Y == 0)[0]
    neg_sample_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
    balanced_idx = np.concatenate([pos_idx, neg_sample_idx])
    np.random.shuffle(balanced_idx)
    X_balanced = X[balanced_idx]
    Y_balanced = Y[balanced_idx]

    return X_balanced, Y_balanced


def run(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    balance: bool,
):

    if balance:
        X_train, Y_train = balance_dataset(X_train, Y_train)
        X_test, Y_test = balance_dataset(X_test, Y_test)

    model = EM()
    model.train(X_train, Y_train)
    result = model.test(X_test, Y_test)

    print(f"Result: {result}")


if __name__ == "__main__":

    data = pd.read_csv("speeddating.csv")

    X_list = []
    for f in ["attractive", "sincere", "intelligence"]:
        columns = get_columns(f)
        feature = Feature(f, columns, data[columns])
        feature.encode()
        X_list.append(feature.encoded_data)
    X = np.column_stack(X_list)
    Y = np.array(data["match"])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=True
    )

    run(X_train, Y_train, X_test, Y_test, balance=True)
