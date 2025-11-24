import numpy as np
from Model import Model


class ML(Model):

    Y_given_X: dict[tuple[int], np.floating]
    random_seed: int

    def __init__(self, random_seed=42) -> None:
        self.Y_given_X = dict()
        self.random_seed = random_seed

    def train(self, X: np.ndarray, Y: np.ndarray, verbose=False) -> None:
        X_tuples = [tuple(row) for row in X]
        unique_rows, inverse_indices = np.unique(X_tuples, return_inverse=True, axis=0)
        unique_rows_tuples = [tuple(row) for row in unique_rows]

        for i, row in enumerate(unique_rows_tuples):
            self.Y_given_X[row] = np.mean(Y[inverse_indices == i])

    def predict(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(self.random_seed)

        X_tuples = [tuple(row) for row in X]
        y = np.empty(len(X), dtype=float)

        for i, x_tuple in enumerate(X_tuples):
            if x_tuple in self.Y_given_X:
                y[i] = self.Y_given_X[x_tuple]
            else:
                y[i] = np.random.rand()

        return y
