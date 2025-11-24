import numpy as np
from Model import Model
from tqdm import tqdm


class EM(Model):

    N_X_STATE: int
    N_Z_STATE: int
    N_X_PER_Z: int
    N_Z: int

    Z_given_X: np.ndarray  # N_Z * (N_X_STATE ** N_X_PER_Z)
    Y_given_Z: np.ndarray  # N_Z_STATE ** N_Z

    Z_given_XY: np.ndarray  # T * (N_Z_STATE ** N_Z)

    def __init__(
        self,
        n_x_state=3,
        n_z_state=2,
        n_x_per_z=5,
        n_z=3,
        random_seed=42,
        init_min=0.3,
        init_max=0.7,
    ) -> None:

        self.N_X_STATE = n_x_state
        self.N_Z_STATE = n_z_state
        self.N_X_PER_Z = n_x_per_z
        self.N_Z = n_z

        np.random.seed(random_seed)
        self.Z_given_X = np.random.uniform(
            init_min, init_max, (self.N_Z, self.N_X_STATE**self.N_X_PER_Z)
        )
        self.Y_given_Z = np.random.uniform(init_min, init_max, self.N_Z_STATE**self.N_Z)

    def estimate(self, X: np.ndarray, Y: np.ndarray) -> None:

        z = np.arange(self.N_Z_STATE**self.N_Z)
        Z_bits = [(z >> i) & 1 for i in range(self.N_Z)]
        Y = Y.reshape(-1, 1)
        self.Z_given_XY = np.where(Y == 1, self.Y_given_Z, 1 - self.Y_given_Z)

        for i in range(self.N_Z):
            pZ = self.Z_given_X[i][X[:, i]]
            Z = Z_bits[i]
            pZ_X = Z * pZ[:, None] + (1 - Z) * (1 - pZ[:, None])
            self.Z_given_XY *= pZ_X

        self.Z_given_XY /= self.Z_given_XY.sum(axis=1, keepdims=True)

    def update(self, X: np.ndarray, Y: np.ndarray) -> None:

        Y_mask = Y == 1
        numerator = (self.Z_given_XY[Y_mask]).sum(axis=0)
        denominator = self.Z_given_XY.sum(axis=0)
        self.Y_given_Z = numerator / denominator

        for i in range(self.N_Z):

            Xi_onehot = X[:, i][:, None] == np.arange(self.N_X_STATE**self.N_X_PER_Z)
            z = np.arange(self.N_Z_STATE**self.N_Z)
            Zi = (z >> i) & 1
            numerator = (Xi_onehot[:, :, None] * self.Z_given_XY[:, None, :] * Zi).sum(
                axis=(0, 2)
            )
            denominator = (Xi_onehot[:, :, None] * self.Z_given_XY[:, None, :]).sum(
                axis=(0, 2)
            )
            nonzero_mask = denominator != 0
            self.Z_given_X[i][nonzero_mask] = (
                numerator[nonzero_mask] / denominator[nonzero_mask]
            )

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        verbose=False,
        max_epochs=100,
        diff_ll=1e-2,
        clip=1e-12,
    ) -> None:
        prev_log_likelihood = -np.inf

        for epoch in tqdm(range(max_epochs)):
            self.estimate(X, Y)
            prediction = self.predict(X)
            prediction_clipped = np.clip(prediction, clip, 1 - clip)
            log_likelihood = np.sum(
                Y * np.log(prediction_clipped)
                + (1 - Y) * np.log(1 - prediction_clipped)
            )
            if verbose:
                print(f"Epoch {epoch+1}, log-likelihood: {log_likelihood:.6f}")
            self.update(X, Y)

            if verbose and abs(log_likelihood - prev_log_likelihood) < diff_ll:
                print("Early stop due to convergency")
                break
            prev_log_likelihood = log_likelihood

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = np.arange(self.N_Z_STATE**self.N_Z)
        Z_bits = [(z >> i) & 1 for i in range(self.N_Z)]
        prob_Z_given_X = np.ones((len(X), self.N_Z_STATE**self.N_Z))

        for i in range(self.N_Z):
            pZ = self.Z_given_X[i][X[:, i]]
            Zi = Z_bits[i]
            pX_given_Z = Zi * pZ[:, None] + (1 - Zi) * (1 - pZ[:, None])
            prob_Z_given_X *= pX_given_Z

        prob_Z_given_X /= prob_Z_given_X.sum(axis=1, keepdims=True)

        return np.dot(prob_Z_given_X, self.Y_given_Z)
