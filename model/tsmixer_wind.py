"""
TSMixer for Wind Production Forecasting
========================================
Based on: "TSMixer: An All-MLP Architecture for Time Series Forecasting"
          (Chen et al., 2023) — https://arxiv.org/abs/2303.06053

Your setup:
  - 16 location files  →  F features total (e.g. 16 × 6 = 96)
  - 1 global production target
  - Input:  (batch, lookback, F)
  - Output: (batch, horizon)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
#  BUILDING BLOCKS
# ─────────────────────────────────────────────

class ResidualNorm(nn.Module):
    """Pre-norm residual wrapper: LayerNorm → block → residual add."""
    def __init__(self, dim: int, block: nn.Module):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.block = block

    def forward(self, x):
        # x: (batch, time, features)
        return x + self.block(self.norm(x))


class TimeMixing(nn.Module):
    """
    Mixes information ACROSS TIME for each feature independently.
    Operates on the time dimension (dim=1).

    Input/output: (batch, time, features)
    """
    def __init__(self, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.fc      = nn.Linear(seq_len, seq_len)
        self.act     = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, F)
        # Transpose → (B, F, T), mix time, transpose back → (B, T, F)
        x = x.transpose(1, 2)          # (B, F, T)
        x = self.dropout(self.act(self.fc(x)))  # (B, F, T)
        return x.transpose(1, 2)       # (B, T, F)


class FeatureMixing(nn.Module):
    """
    Mixes information ACROSS FEATURES at each time step.
    A two-layer MLP applied position-wise.

    Input/output: (batch, time, features)
    """
    def __init__(self, n_features: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = n_features * expansion
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_features),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)   # (B, T, F) → (B, T, F)


class TSMixerBlock(nn.Module):
    """
    One TSMixer block = TimeMixing + FeatureMixing, both with residual connections.
    """
    def __init__(self, seq_len: int, n_features: int,
                 expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.time_mix    = ResidualNorm(
            dim   = n_features,
            block = TimeMixing(seq_len, dropout)
        )
        self.feature_mix = ResidualNorm(
            dim   = n_features,
            block = FeatureMixing(n_features, expansion, dropout)
        )

    def forward(self, x):
        x = self.time_mix(x)
        x = self.feature_mix(x)
        return x


# ─────────────────────────────────────────────
#  FULL TSMIXER MODEL
# ─────────────────────────────────────────────

class TSMixer(nn.Module):
    """
    TSMixer for multivariate → univariate forecasting.

    Args:
        seq_len    : lookback window length (e.g. 48)
        horizon    : forecast steps (e.g. 24)
        n_features : total number of input features (e.g. 96 for 16 locs × 6 vars)
        n_blocks   : number of stacked TSMixer blocks (default 4)
        expansion  : MLP hidden dim multiplier in FeatureMixing (default 4)
        dropout    : dropout probability (default 0.1)

    Input  shape: (batch, seq_len, n_features)
    Output shape: (batch, horizon)
    """
    def __init__(
        self,
        seq_len    : int,
        horizon    : int,
        n_features : int,
        n_blocks   : int   = 4,
        expansion  : int   = 4,
        dropout    : float = 0.1,
    ):
        super().__init__()

        self.seq_len    = seq_len
        self.horizon    = horizon
        self.n_features = n_features

        # Stack of TSMixer blocks
        self.blocks = nn.Sequential(*[
            TSMixerBlock(seq_len, n_features, expansion, dropout)
            for _ in range(n_blocks)
        ])

        # Final projection: flatten (seq_len × n_features) → horizon
        self.head = nn.Linear(seq_len * n_features, horizon)

    def forward(self, x):
        # x: (B, T, F)
        x = self.blocks(x)               # (B, T, F)
        x = x.flatten(start_dim=1)       # (B, T*F)
        return self.head(x)              # (B, horizon)


# ─────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_tsmixer(
    model,
    train_loader,
    val_loader,
    epochs       : int   = 100,
    lr           : float = 1e-3,
    weight_decay : float = 1e-4,
    patience     : int   = 10,
    device       : str   = "auto",
):
    """
    Full training loop with validation, early stopping, and LR scheduling.

    Returns
    -------
    history : dict with 'train_loss' and 'val_loss' lists
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Training on: {device}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion    = nn.MSELoss()
    early_stop   = EarlyStopping(patience=patience)
    history      = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):

        # ── Train ──────────────────────────────
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ───────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train={train_loss:.4f}  val={val_loss:.4f}"
                  f"  lr={optimizer.param_groups[0]['lr']:.2e}")

        if early_stop.step(val_loss, model):
            print(f"\nEarly stopping at epoch {epoch}. Best val loss: {early_stop.best_loss:.4f}")
            early_stop.restore_best(model)
            break

    return history


# ─────────────────────────────────────────────
#  EVALUATION HELPERS
# ─────────────────────────────────────────────

# def evaluate(model, test_loader, scaler_y, device="auto"):
#     """
#     Run inference on the test set, inverse-transform predictions,
#     and compute MAE / RMSE / MAPE.
#     """
#     if device == "auto":
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.eval().to(device)

#     preds_all, trues_all = [], []
#     with torch.no_grad():
#         for X_batch, y_batch in test_loader:
#             pred = model(X_batch.to(device)).cpu().numpy()
#             preds_all.append(pred)
#             trues_all.append(y_batch.numpy())

#     preds = np.concatenate(preds_all)   # (N, horizon)
#     trues = np.concatenate(trues_all)   # (N, horizon)

#     # Inverse scale — scaler was fit on shape (n, 1)
#     preds_inv = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
#     trues_inv = scaler_y.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)

#     mae  = np.mean(np.abs(preds_inv - trues_inv))
#     rmse = np.sqrt(np.mean((preds_inv - trues_inv) ** 2))
#     mask = trues_inv != 0
#     mape = np.mean(np.abs((preds_inv[mask] - trues_inv[mask]) / trues_inv[mask])) * 100

#     print(f"\n{'─'*35}")
#     print(f"  MAE  : {mae:.4f}")
#     print(f"  RMSE : {rmse:.4f}")
#     print(f"  MAPE : {mape:.2f}%")
#     print(f"{'─'*35}\n")

#     return preds_inv, trues_inv, {"mae": mae, "rmse": rmse, "mape": mape}
def evaluate(model, test_loader, scaler_y=None, device="auto"):
    """
    Run inference on the test set and compute MAE / RMSE / MAPE.

    If scaler_y is provided, predictions and targets are inverse-transformed
    before computing metrics.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval().to(device)

    preds_all, trues_all = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(device)).cpu().numpy()

            preds_all.append(pred)
            trues_all.append(y_batch.numpy())

    preds = np.concatenate(preds_all)   # (N, horizon)
    trues = np.concatenate(trues_all)   # (N, horizon)

    # Inverse scale only if a scaler is provided
    if scaler_y is not None:
        preds = scaler_y.inverse_transform(
            preds.reshape(-1, 1)
        ).reshape(preds.shape)

        trues = scaler_y.inverse_transform(
            trues.reshape(-1, 1)
        ).reshape(trues.shape)

    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2))

    mask = trues != 0
    mape = np.mean(
        np.abs((preds[mask] - trues[mask]) / trues[mask])
    ) * 100

    print(f"\n{'─'*35}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"{'─'*35}\n")

    return preds, trues, {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }




# ─────────────────────────────────────────────
#  QUICK-START  (replace with your real data)
# ─────────────────────────────────────────────

def to_loader(X, y, batch_size=32, shuffle=False):
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


from sklearn.preprocessing import StandardScaler

datetime_cols = [
 'hour_sin',
 'hour_cos',
 'day_sin',
 'day_cos',
 'day_of_the_year_sin',
 'day_of_the_year_cos',
 'month_sin',
 'month_cos',
 'quarter_sin',
 'quarter_cos']

def create_sequences(data, target_name, lookback=48, horizon=24):
    """
    lookback : how many past timesteps the model sees
    horizon  : how many future timesteps to predict
    """
    met_cols = [col for col in data.columns if col not in datetime_cols + [target_name]]

    df_out = data.copy()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    df_out[met_cols] = scaler_X.fit_transform(df_out[met_cols])
    df_out[target_name] = scaler_y.fit_transform(df_out[[target_name]])

    features_cols = met_cols + datetime_cols
    X_raw = df_out[features_cols].values         # (N, F)
    y_raw = df_out[[target_name]].values

    length = X_raw.shape[0]

    X_list, y_list = [], []

    for i in range(length - lookback - horizon + 1):
        x_window = X_raw[i : i + lookback]           # (lookback, features)
        y_window = y_raw[i + lookback : i + lookback + horizon]  # (horizon,)
        X_list.append(x_window)
        y_list.append(y_window)

    return np.array(X_list), np.array(y_list)

def to_loader(X, y,val_size,test_size, batch_size=32, shuffle=False):
    n=len(X)
    y=y.squeeze() # (N, horizon,1) → (N,horizon)
    train_end = n-val_size-test_size
    val_end = n-test_size
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Val:   {X_val.shape}, {y_val.shape}")
    print(f"Test:  {X_test.shape}, {y_test.shape}")
    dataset1 = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    dataset2 = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    dataset3 = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    return DataLoader(dataset1, batch_size=batch_size, shuffle=shuffle), DataLoader(dataset2, batch_size=batch_size, shuffle=shuffle), DataLoader(dataset3, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":


    LOOKBACK = 24   # how many past steps the model sees
    HORIZON  = 24   # how many future steps to predict

    # ── 1. Data ──────────────────────────────────────────────────────────
    import pandas as pd
    data = pd.read_csv("./dane/dane_wind.csv", parse_dates=['Date'],index_col='Date')

    target_name = "fw_production" #fw_prod_diff jeśli dane_wind_diff.csv
    X, y = create_sequences(data, target_name, lookback=LOOKBACK, horizon=HORIZON)

    train_loader, val_loader, test_loader = to_loader(X, y, val_size=168, test_size=24, batch_size=32, shuffle=True)
    
    # ── 2. Model ─────────────────────────────────────────────────────────
    N_FEATURES = X.shape[2]  

    model = TSMixer(
        seq_len    = LOOKBACK,
        horizon    = HORIZON,
        n_features = N_FEATURES,
        n_blocks   = 4,       # increase for more capacity
        expansion  = 4,       # MLP hidden = n_features * expansion
        dropout    = 0.1,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(model)

    # ── 3. Train ─────────────────────────────────────────────────────────
    history = train_tsmixer(
        model,
        train_loader,
        val_loader,
        epochs       = 100,
        lr           = 1e-3,
        weight_decay = 1e-4,
        patience     = 15,
    )
    
    # ── 4. Evaluate ───────────────────────────────────────────────────────
    preds, trues, metrics = evaluate(model, test_loader)

    # ── 5. Visualise ──────────────────────────────────────────────────────
    # plot_results(history, preds, trues)
    
    with open("./wyniki/trains.npy", "wb") as f:
        np.save(f, np.array(history["train_loss"]))
    with open("./wyniki/vals.npy", "wb") as f:
        np.save(f, np.array(history["val_loss"]))
    with open("./wyniki/preds.npy", "wb") as f:
        np.save(f, preds)
    with open("./wyniki/trues.npy", "wb") as f:
        np.save(f, trues)


    # ── 6. Save ───────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "tsmixer_wind.pt")
    print("Model saved to tsmixer_wind.pt")
