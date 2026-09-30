"""Classifier implementations for classical and deep gait recognition."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class Prediction:
    labels: np.ndarray
    confidence: np.ndarray


class ClassicalModel:
    """Thin wrapper exposing a common interface over a scikit-learn estimator."""
    def __init__(self, kind: str = "svm", random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        self.kind = kind.lower()
        if self.kind == "svm":
            estimator = SVC(C=2.0, kernel="rbf", probability=True, random_state=random_state)
        elif self.kind in {"random_forest", "rf"}:
            estimator = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
        elif self.kind in {"knn", "k-nearest-neighbours"}:
            estimator = KNeighborsClassifier(n_neighbors=5, weights="distance")
        else:
            raise ValueError("kind must be svm, random_forest, or knn")
        self.estimator = make_pipeline(StandardScaler(), estimator) if self.kind != "rf" and self.kind != "random_forest" else estimator

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ClassicalModel":
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("non-empty x and matching y are required")
        self.estimator.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        if not hasattr(self.estimator, "classes_") and not hasattr(self.estimator[-1] if hasattr(self.estimator, "__getitem__") else self.estimator, "classes_"):
            raise RuntimeError("model must be fitted before prediction")
        labels = self.estimator.predict(x)
        probabilities = self.estimator.predict_proba(x)
        return Prediction(labels=np.asarray(labels), confidence=np.max(probabilities, axis=1))

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ClassicalModel":
        import joblib
        return joblib.load(path)


class CNNLSTM:
    """Small raw-window CNN-LSTM classifier implemented with PyTorch."""
    def __init__(self, n_classes: int, epochs: int = 8, batch_size: int = 64, seed: int = 42):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("CNN-LSTM requires the optional torch dependency") from exc
        torch.manual_seed(seed)
        self.torch = torch
        self.nn = nn
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=48, batch_first=True)
        self.head = nn.Linear(48, n_classes)
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CNNLSTM":
        torch = self.torch
        self.classes_ = np.unique(y)
        mapping = {label: i for i, label in enumerate(self.classes_)}
        target = np.asarray([mapping[v] for v in y], dtype=np.int64)
        xt = torch.tensor(x, dtype=torch.float32).transpose(1, 2)
        yt = torch.tensor(target, dtype=torch.long)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt, yt), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(list(self.net.parameters()) + list(self.lstm.parameters()) + list(self.head.parameters()), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.net.train(); self.lstm.train(); self.head.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                sequence = self.net(batch_x).transpose(1, 2)
                output, _ = self.lstm(sequence)
                logits = self.head(output[:, -1, :])
                loss_fn(logits, batch_y).backward()
                optimizer.step()
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        if self.classes_ is None:
            raise RuntimeError("model must be fitted before prediction")
        torch = self.torch
        self.net.eval(); self.lstm.eval(); self.head.eval()
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).transpose(1, 2)
            logits = self.head(self.lstm(self.net(xt).transpose(1, 2))[0][:, -1, :])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        indices = np.argmax(probs, axis=1)
        return Prediction(labels=self.classes_[indices], confidence=np.max(probs, axis=1))

    def parameter_count(self) -> int:
        return sum(p.numel() for module in (self.net, self.lstm, self.head) for p in module.parameters())
