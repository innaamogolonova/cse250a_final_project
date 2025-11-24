import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


class Model:

    def __init__(self) -> None:
        raise NotImplementedError

    def train(self, X: np.ndarray, Y: np.ndarray, verbose=False) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def test(self, X: np.ndarray, Y: np.ndarray) -> dict:
        prediction_prob = self.predict(X)
        prediction_label = (prediction_prob >= 0.5).astype(int)

        acc = accuracy_score(Y, prediction_label)
        precision = precision_score(Y, prediction_label)
        recall = recall_score(Y, prediction_label)
        f1 = f1_score(Y, prediction_label)
        roc_auc = roc_auc_score(Y, prediction_prob)
        pr_auc = average_precision_score(Y, prediction_prob)
        cm = confusion_matrix(Y, prediction_label)

        analysis = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
        }

        return analysis
