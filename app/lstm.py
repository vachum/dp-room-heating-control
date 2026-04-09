import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Column order is part of the interface contract between training and inference.
# Any reordering or addition here must be mirrored in controllers.FEATURE_COUNT
# and MPCController._feature().
FEATURE_COLUMNS = ["y", "u", "T_out", "solar", "occupancy", "setpoint", "residual"]
TARGET_COLUMN = "residual_target"


@dataclass
class LSTMTrainResult:
    model_path: str
    meta_path: str
    samples: int
    train_samples: int
    val_samples: int
    train_rmse: float
    val_rmse: float


class ResidualLSTM(nn.Module):
    """Single-layer LSTM that predicts the next-step model residual.

    Input:  [batch, seq_len, n_features]
    Output: [batch] for horizon=1, or [batch, horizon] for multi-step

    Horizon=1 is the production setting.  The predicted scalar is broadcast
    across the full MPC horizon inside MPCController._residual_sequence().
    """

    def __init__(
        self, input_size: int, hidden_size: int = 24, horizon: int = 1
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        # Linear readout from the final hidden state only; earlier time steps
        # contribute through LSTM recurrence, not direct skip connections.
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.head(out[:, -1, :])
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def _build_sequences(
    df: pd.DataFrame, seq_len: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    feat = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    target = df[TARGET_COLUMN].to_numpy(dtype=float)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in range(seq_len - 1, len(df) - horizon + 1):
        xs.append(feat[i - seq_len + 1 : i + 1])
        ys.append(target[i : i + horizon])

    if not xs:
        return np.empty((0, seq_len, len(FEATURE_COLUMNS))), np.empty((0, horizon))

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def train_residual_lstm(
    df: pd.DataFrame,
    out_dir: str,
    seq_len: int = 12,
    horizon: int = 1,
    hidden_size: int = 24,
    epochs: int = 25,
    seed: int = 42,
    metadata: dict | None = None,
) -> LSTMTrainResult:
    missing = [c for c in [*FEATURE_COLUMNS, TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for LSTM training: {missing}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_all, y_all = _build_sequences(
        df.reset_index(drop=True), seq_len=seq_len, horizon=horizon
    )
    if len(x_all) < 40:
        raise ValueError("Not enough samples for LSTM training")

    # Time-ordered split: keep temporal structure intact.
    # Random shuffling would leak future information into the validation set.
    split_idx = int(len(x_all) * 0.8)
    x_train, x_val = x_all[:split_idx], x_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]

    # Normalization statistics are derived from training data only,
    # then saved to meta.json so inference can reproduce identical scaling.
    feat_mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    feat_std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)  # avoid division by zero for constant features

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-8:
        y_std = 1.0

    x_train_n = (x_train - feat_mean) / feat_std
    x_val_n = (x_val - feat_mean) / feat_std
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    model = ResidualLSTM(
        input_size=len(FEATURE_COLUMNS), hidden_size=hidden_size, horizon=horizon
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x_train_t = torch.tensor(x_train_n, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_n, dtype=torch.float32)

    # Shuffle within the training portion only; the validation tail stays ordered.
    dataset = TensorDataset(x_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in dataloader:
            opt.zero_grad()
            pred = model(batch_x)
            # Align shapes: ResidualLSTM squeezes the last dim for horizon=1,
            # but the target tensor still has shape [batch, 1] from _build_sequences.
            if pred.ndim == 1 and batch_y.ndim == 2 and batch_y.shape[1] == 1:
                pred = pred.unsqueeze(-1)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_train = model(torch.tensor(x_train_n, dtype=torch.float32)).cpu().numpy()
        pred_val = model(torch.tensor(x_val_n, dtype=torch.float32)).cpu().numpy()

    train_rmse = float(np.sqrt(np.mean(((pred_train * y_std + y_mean) - y_train) ** 2)))
    val_rmse = float(np.sqrt(np.mean(((pred_val * y_std + y_mean) - y_val) ** 2)))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_path = out_path / "residual_lstm.pt"
    meta_path = out_path / "residual_lstm_meta.json"

    torch.save(model.state_dict(), model_path)
    meta = {
        "split_strategy": "time_ordered",
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "seq_len": seq_len,
        "horizon": horizon,
        "horizon_steps": horizon,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "feature_columns": FEATURE_COLUMNS,
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "target_mean": y_mean,
        "target_std": y_std,
    }
    if metadata:
        meta.update(metadata)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    return LSTMTrainResult(
        model_path=str(model_path),
        meta_path=str(meta_path),
        samples=len(x_all),
        train_samples=len(x_train),
        val_samples=len(x_val),
        train_rmse=train_rmse,
        val_rmse=val_rmse,
    )


class ResidualLSTMPredictor:
    def __init__(
        self,
        model: ResidualLSTM,
        seq_len: int,
        horizon: int,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        target_mean: float,
        target_std: float,
    ) -> None:
        self.model = model
        self.seq_len = seq_len
        self.horizon = horizon
        self.feat_mean = feat_mean
        self.feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std if abs(target_std) > 1e-8 else 1.0)

        self._history: deque[np.ndarray] = deque(maxlen=seq_len)
        self._bootstrap_cache: list[np.ndarray] = []

    def bootstrap(self, history_df: pd.DataFrame) -> None:
        """Pre-fill the rolling history from a warm-up trace.

        Called once after the warm-up phase so the first real forecast has
        a full seq_len of context rather than zero-padded entries.
        The snapshot is saved so reset() can restore this state between runs.
        """
        self._history.clear()
        for _, row in history_df.tail(self.seq_len).iterrows():
            feat = row[FEATURE_COLUMNS].to_numpy(dtype=float)
            self._history.append(feat)
        self._bootstrap_cache = list(self._history)

    def reset(self) -> None:
        # Restore to post-bootstrap state so each controller reset() starts
        # with the same warm history across scenario reruns.
        self._history.clear()
        for feat in self._bootstrap_cache:
            self._history.append(feat)

    def _prepare_seq(self, current_feature: np.ndarray) -> np.ndarray:
        seq = list(self._history)
        seq.append(np.asarray(current_feature, dtype=float))

        if not seq:
            seq = [np.asarray(current_feature, dtype=float)]

        # Pad by repeating the oldest frame when history is shorter than seq_len.
        # This happens at the very start of a run before the deque fills up.
        while len(seq) < self.seq_len:
            seq.insert(0, seq[0])

        return np.asarray(seq[-self.seq_len :], dtype=float)

    def forecast(self, horizon: int, current_feature: np.ndarray) -> np.ndarray:
        seq = self._prepare_seq(current_feature)
        seq_n = (seq - self.feat_mean) / self.feat_std

        x = torch.tensor(seq_n[None, :, :], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            pred_n = self.model(x).cpu().numpy().flatten()

        pred = pred_n * self.target_std + self.target_mean

        if len(pred) == 1 and horizon > 1:
            logger.warning(
                "LSTM returned 1 value instead of horizon=%d. "
                "Broadcasting scalar — check horizon in model config.",
                horizon,
            )
            return np.full(int(horizon), pred[0], dtype=float)

        if len(pred) >= horizon:
            return pred[:horizon]
        else:
            logger.warning(
                "LSTM output length %d < horizon %d. Padding with last value.",
                len(pred),
                horizon,
            )
            return np.pad(pred, (0, horizon - len(pred)), mode="edge")

    def update(self, feature: np.ndarray, residual: float) -> None:
        self._history.append(np.asarray(feature, dtype=float))


def load_residual_predictor(
    model_dir: str,
) -> ResidualLSTMPredictor:
    base = Path(model_dir)
    model_path = base / "residual_lstm.pt"
    meta_path = base / "residual_lstm_meta.json"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = ResidualLSTM(
        input_size=len(meta["feature_columns"]),
        hidden_size=int(meta["hidden_size"]),
        horizon=int(meta.get("horizon", 1)),
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    predictor = ResidualLSTMPredictor(
        model=model,
        seq_len=int(meta["seq_len"]),
        horizon=int(meta.get("horizon", 1)),
        feat_mean=np.asarray(meta["feat_mean"], dtype=float),
        feat_std=np.asarray(meta["feat_std"], dtype=float),
        target_mean=float(meta["target_mean"]),
        target_std=float(meta["target_std"]),
    )
    return predictor
