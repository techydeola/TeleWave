"""
Standard Recurrent LSTM baseline forecaster for raw meteorological time-series.
Features recurrent hidden memory cells with regularized linear projection.
"""

import math
import random
from typing import List, Dict, Any, Tuple
from climatewave.models.base import BaseClimateForecaster


def sigmoid(x: float) -> float:
    x_clipped = max(-15.0, min(15.0, x))
    return 1.0 / (1.0 + math.exp(-x_clipped))


def tanh(x: float) -> float:
    x_clipped = max(-15.0, min(15.0, x))
    return math.tanh(x_clipped)


def solve_ridge_regression(X_mat: List[List[float]], y_vec: List[float], lambda_reg: float = 1e-4) -> List[float]:
    """Solves w = (X^T X + lambda*I)^(-1) X^T y using Gaussian elimination."""
    n_samples = len(X_mat)
    n_features = len(X_mat[0])

    # Compute XtX
    XtX = [[0.0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        for j in range(n_features):
            s = sum(X_mat[k][i] * X_mat[k][j] for k in range(n_samples))
            if i == j:
                s += lambda_reg
            XtX[i][j] = s

    # Compute Xty
    Xty = [sum(X_mat[k][i] * y_vec[k] for k in range(n_samples)) for i in range(n_features)]

    # Gaussian elimination with partial pivoting
    n = n_features
    M = [XtX[i][:] + [Xty[i]] for i in range(n)]
    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[max_row][i]):
                max_row = k
        M[i], M[max_row] = M[max_row], M[i]

        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            pivot = 1e-12 if pivot >= 0 else -1e-12

        for j in range(i, n + 1):
            M[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(i, n + 1):
                    M[k][j] -= factor * M[i][j]

    return [M[i][n] for i in range(n)]


class BaselineLSTMForecaster(BaseClimateForecaster):
    """
    Standard Single-Branch LSTM Baseline for time-series forecasting.
    """

    def __init__(
        self,
        name: str = "Standard LSTM",
        hidden_dim: int = 12,
        learning_rate: float = 0.02,
        seed: int = 42,
    ):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.seed = seed
        random.seed(seed)

        self.input_dim = 1
        self.output_dim = 1
        self.lookback = 12
        self.weights: Dict[str, Any] = {}
        self.proj_weights: List[float] = []

    def _init_weights(self, input_dim: int, output_dim: int, lookback: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lookback = lookback
        H = self.hidden_dim
        I = input_dim

        std_in = math.sqrt(2.0 / (I + H))
        std_h = math.sqrt(2.0 / (H + H))

        for gate in ["i", "f", "c", "o"]:
            self.weights[f"W_{gate}"] = [[random.gauss(0, std_in * 0.4) for _ in range(H)] for _ in range(I)]
            self.weights[f"U_{gate}"] = [[random.gauss(0, std_h * 0.4) for _ in range(H)] for _ in range(H)]
            self.weights[f"b_{gate}"] = [0.0 if gate != "f" else 1.0 for _ in range(H)]

    def _extract_hidden_features(self, seq: List[List[float]]) -> List[float]:
        H = self.hidden_dim
        I = self.input_dim

        h = [0.0] * H
        c = [0.0] * H

        for x_t in seq:
            h_next = [0.0] * H
            c_next = [0.0] * H
            for j in range(H):
                z_i = sum(x_t[k] * self.weights["W_i"][k][j] for k in range(I)) + sum(h[k] * self.weights["U_i"][k][j] for k in range(H)) + self.weights["b_i"][j]
                z_f = sum(x_t[k] * self.weights["W_f"][k][j] for k in range(I)) + sum(h[k] * self.weights["U_f"][k][j] for k in range(H)) + self.weights["b_f"][j]
                z_c = sum(x_t[k] * self.weights["W_c"][k][j] for k in range(I)) + sum(h[k] * self.weights["U_c"][k][j] for k in range(H)) + self.weights["b_c"][j]
                z_o = sum(x_t[k] * self.weights["W_o"][k][j] for k in range(I)) + sum(h[k] * self.weights["U_o"][k][j] for k in range(H)) + self.weights["b_o"][j]

                i_g = sigmoid(z_i)
                f_g = sigmoid(z_f)
                c_tilde = tanh(z_c)
                c_next[j] = f_g * c[j] + i_g * c_tilde
                h_next[j] = sigmoid(z_o) * tanh(c_next[j])
            h, c = h_next, c_next

        # Feature vector: [1.0 (bias), target lag 0..T, hidden state 0..H]
        feat = [1.0]
        for step in seq:
            feat.append(step[0])
        feat.extend(h)
        return feat

    def fit(
        self,
        X_train: List[List[List[float]]],
        y_train: List[List[float]],
        epochs: int = 20,
        batch_size: int = 16,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        if not X_train:
            return {"loss_history": []}

        input_dim = len(X_train[0][0])
        output_dim = len(y_train[0])
        lookback = len(X_train[0])
        self._init_weights(input_dim, output_dim, lookback)

        # Extract features for all samples
        H_mat = [self._extract_hidden_features(seq) for seq in X_train]
        y_vec = [y[0] for y in y_train]

        # Solve optimal regularized projection
        self.proj_weights = solve_ridge_regression(H_mat, y_vec, lambda_reg=0.01)

        # Calculate train loss
        preds = [sum(h_row[j] * self.proj_weights[j] for j in range(len(self.proj_weights))) for h_row in H_mat]
        mse = sum((p - y) ** 2 for p, y in zip(preds, y_vec)) / len(y_vec)

        self.is_fitted = True
        return {"final_loss": round(mse, 5), "loss_history": [round(mse, 5)]}

    def predict(self, X: List[List[List[float]]]) -> List[List[float]]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict().")

        preds = []
        for seq in X:
            feat = self._extract_hidden_features(seq)
            out_val = sum(feat[j] * self.proj_weights[j] for j in range(len(self.proj_weights)))
            preds.append([out_val])
        return preds
