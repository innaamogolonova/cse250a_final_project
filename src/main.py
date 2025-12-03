import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Feature import Feature
from Model import Model
from EM import EM
from ML import ML

import data_info as di


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

    return X[balanced_idx], Y[balanced_idx]


def run(
    model: Model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    balance: bool,
    show_data_info: bool,
    show_plot: bool,
):

    if show_data_info:
        df = di.get_feature_df()
        di.get_data_info(df,show_plot)

    if balance:
        X_train, Y_train = balance_dataset(X_train, Y_train)
        X_test, Y_test = balance_dataset(X_test, Y_test)

    model.train(X_train, Y_train)
    result = model.test(X_test, Y_test)

    print(f"Result: {result}")


if __name__ == "__main__":

    data = pd.read_csv("../speeddating.csv")

    X_list = []
    for f in ["attractive", "sincere", "intelligence"]:
        # columns = get_columns(f)
        columns = di.get_features(f,True)
        feature = Feature(f, columns, data[columns])
        feature.encode()
        X_list.append(feature.encoded_data)
    X = np.column_stack(X_list)
    Y = np.array(data["match"])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=True
    )

    run(EM(), X_train, Y_train, X_test, Y_test, balance=False, show_data_info=False, show_plot=False)
    run(EM(), X_train, Y_train, X_test, Y_test, balance=True, show_data_info=False, show_plot=False)
    run(ML(), X_train, Y_train, X_test, Y_test, balance=False, show_data_info=False, show_plot=False)
    run(ML(), X_train, Y_train, X_test, Y_test, balance=True, show_data_info=False, show_plot=False)



    data_2 = di.get_feature_df()

    X_2 = data_2.iloc[:,:15].copy().to_numpy()
    Y_2 = np.array(data_2["match"].copy())

    X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(
    X_2, Y_2, test_size=0.2, random_state=42, shuffle=True
    )


    run(EM(), X_train_2, Y_train_2, X_test_2, Y_test_2, balance=False, show_data_info=False, show_plot=False)
    run(EM(), X_train_2, Y_train_2, X_test_2, Y_test_2, balance=True, show_data_info=False, show_plot=False)
    run(ML(), X_train_2, Y_train_2, X_test_2, Y_test_2, balance=False, show_data_info=False, show_plot=False)
    run(ML(), X_train_2, Y_train_2, X_test_2, Y_test_2, balance=True, show_data_info=False, show_plot=False)


