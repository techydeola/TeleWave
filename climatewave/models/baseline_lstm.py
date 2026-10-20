"""
Standard Recurrent LSTM baseline forecaster for raw meteorological time-series.
Features recurrent hidden memory cells with regularized linear projection, vectorized with NumPy.
"""

import math
import random
from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from climatewave.models.base import BaseClimateForecaster


def sigmoid(x: float) -> float:
    x_clipped = max(-15.0, min(15.0, x))
    return 1.0 / (1.0 + math.exp(-x_clipped))


def tanh(x: float) -> float:
    x_clipped = max(-15.0, min(15.0, x))
    return math.tanh(x_clipped)


def solve_ridge_regression(
    X_mat: Union[List[List[float]], np.ndarray],
    y_vec: Union[List[float], np.ndarray],
    lambda_reg: float = 1e-4,
) -> List[float]:
    """
    Solves w = (X^T X + lambda*I)^(-1) X^T y using high-performance NumPy linear algebra.
    """
    X = np.asarray(X_mat, dtype=np.float64)
    y = np.asarray(y_vec, dtype=np.float64)
    n_features = X.shape[1]

    XtX = np.dot(X.T, X)
    XtX.flat[:: n_features + 1] += lambda_reg
    Xty = np.dot(X.T, y)

    w = np.linalg.solve(XtX, Xty)
    return w.tolist()


class BaselineLSTMForecaster(BaseClimateForecaster):
    """
    Standard Single-Branch LSTM Baseline for time-series forecasting.
    Vectorized batch recurrence with regularized closed-form linear projection.
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

        self.input_dim = 1
        self.output_dim = 1
        self.lookback = 12
        self.weights: Dict[str, Any] = {}
        self.proj_weights: List[float] = []

        # Packed NumPy arrays for vectorized recurrence
        self._W: Optional[np.ndarray] = None
        self._U: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None
        self._last_packed_weights_id: Optional[int] = None

    def _init_weights(self, input_dim: int, output_dim: int, lookback: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lookback = lookback
        H = self.hidden_dim
        I = input_dim

        std_in = math.sqrt(2.0 / (I + H))
        std_h = math.sqrt(2.0 / (H + H))

        random.seed(self.seed)
        for gate in ["i", "f", "c", "o"]:
            self.weights[f"W_{gate}"] = [[random.gauss(0, std_in * 0.4) for _ in range(H)] for _ in range(I)]
            self.weights[f"U_{gate}"] = [[random.gauss(0, std_h * 0.4) for _ in range(H)] for _ in range(H)]
            self.weights[f"b_{gate}"] = [0.0 if gate != "f" else 1.0 for _ in range(H)]

        self._pack_weights()

    def _pack_weights(self):
        """Packs gate weights dictionary into contiguous NumPy arrays for vectorized recurrence."""
        self._W = np.column_stack([
            np.asarray(self.weights["W_i"], dtype=np.float64),
            np.asarray(self.weights["W_f"], dtype=np.float64),
            np.asarray(self.weights["W_c"], dtype=np.float64),
            np.asarray(self.weights["W_o"], dtype=np.float64),
        ])
        self._U = np.column_stack([
            np.asarray(self.weights["U_i"], dtype=np.float64),
            np.asarray(self.weights["U_f"], dtype=np.float64),
            np.asarray(self.weights["U_c"], dtype=np.float64),
            np.asarray(self.weights["U_o"], dtype=np.float64),
        ])
        self._b = np.concatenate([
            np.asarray(self.weights["b_i"], dtype=np.float64),
            np.asarray(self.weights["b_f"], dtype=np.float64),
            np.asarray(self.weights["b_c"], dtype=np.float64),
            np.asarray(self.weights["b_o"], dtype=np.float64),
        ])
        self._last_packed_weights_id = id(self.weights)

    def _ensure_packed_weights(self):
        if self._W is None or self._last_packed_weights_id != id(self.weights):
            self._pack_weights()

    def _extract_hidden_features(self, seq: Union[List[List[float]], np.ndarray]) -> List[float]:
        self._ensure_packed_weights()
        seq_arr = np.asarray(seq, dtype=np.float64)
        T = len(seq_arr)
        H = self.hidden_dim
        h = np.zeros(H, dtype=np.float64)
        c = np.zeros(H, dtype=np.float64)

        for t in range(T):
            gates = np.dot(seq_arr[t], self._W) + np.dot(h, self._U) + self._b
            z_i, z_f, z_c, z_o = np.split(gates, 4)
            i_g = 1.0 / (1.0 + np.exp(-np.clip(z_i, -15.0, 15.0)))
            f_g = 1.0 / (1.0 + np.exp(-np.clip(z_f, -15.0, 15.0)))
            c_tilde = np.tanh(np.clip(z_c, -15.0, 15.0))
            c = f_g * c + i_g * c_tilde
            h = (1.0 / (1.0 + np.exp(-np.clip(z_o, -15.0, 15.0)))) * np.tanh(np.clip(c, -15.0, 15.0))

        feat = [1.0] + seq_arr[:, 0].tolist() + h.tolist()
        return feat

    def _extract_batch_hidden_features(self, X: Union[List[List[List[float]]], np.ndarray]) -> np.ndarray:
        self._ensure_packed_weights()
        X_arr = np.asarray(X, dtype=np.float64)
        N, T, _ = X_arr.shape
        H = self.hidden_dim
        h = np.zeros((N, H), dtype=np.float64)
        c = np.zeros((N, H), dtype=np.float64)

        for t in range(T):
            gates = np.dot(X_arr[:, t, :], self._W) + np.dot(h, self._U) + self._b
            z_i, z_f, z_c, z_o = np.split(gates, 4, axis=1)
            i_g = 1.0 / (1.0 + np.exp(-np.clip(z_i, -15.0, 15.0)))
            f_g = 1.0 / (1.0 + np.exp(-np.clip(z_f, -15.0, 15.0)))
            c_tilde = np.tanh(np.clip(z_c, -15.0, 15.0))
            c = f_g * c + i_g * c_tilde
            h = (1.0 / (1.0 + np.exp(-np.clip(z_o, -15.0, 15.0)))) * np.tanh(np.clip(c, -15.0, 15.0))

        bias = np.ones((N, 1), dtype=np.float64)
        lags = X_arr[:, :, 0]
        return np.column_stack([bias, lags, h])

    def fit(
        self,
        X_train: Union[List[List[List[float]]], np.ndarray],
        y_train: Union[List[List[float]], np.ndarray],
        epochs: int = 20,
        batch_size: int = 16,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        if len(X_train) == 0:
            return {"loss_history": []}

        input_dim = len(X_train[0][0])
        output_dim = len(y_train[0])
        lookback = len(X_train[0])
        self._init_weights(input_dim, output_dim, lookback)

        # Batch-vectorized feature extraction
        H_mat = self._extract_batch_hidden_features(X_train)
        y_vec = np.asarray(y_train, dtype=np.float64)[:, 0]

        # Solve optimal regularized projection
        self.proj_weights = solve_ridge_regression(H_mat, y_vec, lambda_reg=0.01)

        # Calculate train loss
        preds = np.dot(H_mat, self.proj_weights)
        mse = float(np.mean((preds - y_vec) ** 2))

        self.is_fitted = True
        return {"final_loss": round(mse, 5), "loss_history": [round(mse, 5)]}

    def predict(self, X: Union[List[List[List[float]]], np.ndarray]) -> List[List[float]]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict().")

        H_mat = self._extract_batch_hidden_features(X)
        preds = np.dot(H_mat, self.proj_weights)
        return [[float(p)] for p in preds]
