"""
Train LSTM on real Backblaze Q1 2020 data.
Maps Backblaze SMART attributes to the project's schema:
  read_errors  ← smart_5_raw  (reallocated sectors count)
  write_errors ← smart_187_raw (reported uncorrectable errors)
  temperature  ← smart_194_raw (temperature Celsius, fallback smart_190_raw)
  health_score ← derived from smart_197_raw + smart_198_raw (pending/uncorrectable sectors)

Label: binary failure (0/1) from Backblaze 'failure' column — REAL failure events.

Run from project root:
    python train_lstm_backblaze_final.py
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = "C:/Users/kefii/Downloads/data_Q1_2020/"
MODEL_OUT = "ml-service/models/lstm_model.pth"
SEQ_LEN   = 5
EPOCHS    = 20
BATCH     = 64
MAX_FILES = 7         # use first 7 days — representative sample, fast training
MAX_DISKS = 5000      # limit to 5000 disks max for speed

# Backblaze → project schema mapping
# smart_5   = Reallocated Sectors Count  → read_errors
# smart_187 = Reported Uncorrectable Errors → write_errors
# smart_194 = Temperature Celsius (most drives) → temperature
# smart_190 = Temperature (some drives, fallback)
# health_score = 1 if (smart_197 + smart_198) > 0 else 0
#   smart_197 = Current Pending Sector Count
#   smart_198 = Offline Uncorrectable Sector Count
BB_COLS = [
    "serial_number", "failure",
    "smart_5_raw",    # → read_errors
    "smart_187_raw",  # → write_errors
    "smart_194_raw",  # → temperature (primary)
    "smart_190_raw",  # → temperature (fallback)
    "smart_197_raw",  # → health_score component
    "smart_198_raw",  # → health_score component
]

# Fixed normalization maxima — must match lstm_predictor.py
FEAT_MAX = np.array([100.0, 80.0, 80.0, 1.0], dtype=np.float32)

print("=" * 60)
print("  LSTM Training on Backblaze Q1 2020")
print("=" * 60)

# ── 1. Load data ──────────────────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))[:MAX_FILES]
print(f"\nLoading {len(csv_files)} CSV files...")

frames = []
for f in csv_files:
    try:
        df = pd.read_csv(f, usecols=BB_COLS)
        frames.append(df)
    except Exception as e:
        print(f"  Skipping {os.path.basename(f)}: {e}")

raw = pd.concat(frames, ignore_index=True)
print(f"Total rows loaded: {len(raw):,}")
print(f"Total failures in dataset: {raw['failure'].sum()}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
raw = raw.fillna(0)

# Map to project schema
raw["read_errors"]  = raw["smart_5_raw"].clip(0, 100)
raw["write_errors"] = raw["smart_187_raw"].clip(0, 80)

# Temperature: prefer smart_194, fallback to smart_190
raw["temperature"] = np.where(
    raw["smart_194_raw"] > 0,
    raw["smart_194_raw"],
    raw["smart_190_raw"]
).clip(0, 80)

# health_score: 0.0 = healthy, 0.5 = warning (pending sectors), 1.0 = critical
pending     = raw["smart_197_raw"]
uncorrect   = raw["smart_198_raw"]
raw["health_score"] = np.where(
    uncorrect > 0, 1.0,
    np.where(pending > 0, 0.5, 0.0)
).astype(np.float32)

FEATURES = ["read_errors", "write_errors", "temperature", "health_score"]

print(f"\nFeature stats:")
for f in FEATURES:
    print(f"  {f}: min={raw[f].min():.1f}  max={raw[f].max():.1f}  mean={raw[f].mean():.3f}")

# ── 3. Build per-disk sequences ───────────────────────────────────────────────
print(f"\nBuilding sequences (SEQ_LEN={SEQ_LEN})...")

X_list, y_list = [], []
disks_used = 0

# Group by serial number to keep temporal order per disk
all_serials = raw["serial_number"].unique()
if len(all_serials) > MAX_DISKS:
    np.random.seed(42)
    all_serials = np.random.choice(all_serials, MAX_DISKS, replace=False)
    raw = raw[raw["serial_number"].isin(all_serials)]
    print(f"Sampled {MAX_DISKS} disks from dataset")

for serial, group in raw.groupby("serial_number"):
    group = group.sort_index()  # preserve CSV order (already date-sorted)
    vals = group[FEATURES].values.astype(np.float32)
    labels = group["failure"].values.astype(np.float32)

    if len(vals) < SEQ_LEN + 1:
        continue

    for i in range(len(vals) - SEQ_LEN):
        X_list.append(vals[i:i + SEQ_LEN])
        y_list.append(labels[i + SEQ_LEN])

    disks_used += 1

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f"Disks used: {disks_used:,}")
print(f"Sequences:  {len(X):,}")
print(f"Failures:   {int(y.sum()):,}  ({100*y.mean():.3f}%)")

# ── 4. Normalize ──────────────────────────────────────────────────────────────
X = np.clip(X / FEAT_MAX, 0.0, 1.0)

# ── 5. Handle class imbalance — oversample failures ──────────────────────────
fail_idx    = np.where(y == 1)[0]
normal_idx  = np.where(y == 0)[0]

print(f"\nClass balancing:")
print(f"  Normal sequences:  {len(normal_idx):,}")
print(f"  Failure sequences: {len(fail_idx):,}")

# Oversample failures to 10% of normals
target_failures = max(len(fail_idx), len(normal_idx) // 10)
if len(fail_idx) > 0 and len(fail_idx) < target_failures:
    repeat = int(np.ceil(target_failures / len(fail_idx)))
    fail_idx_oversampled = np.tile(fail_idx, repeat)[:target_failures]
    all_idx = np.concatenate([normal_idx, fail_idx_oversampled])
else:
    all_idx = np.arange(len(X))

np.random.seed(42)
np.random.shuffle(all_idx)
X_bal = X[all_idx]
y_bal = y[all_idx]

print(f"  After balancing: {len(X_bal):,} sequences  ({100*y_bal.mean():.1f}% failures)")

# ── 6. Train/test split (80/20, no shuffle — time-series) ────────────────────
split    = int(len(X_bal) * 0.8)
X_train, X_test = X_bal[:split], X_bal[split:]
y_train, y_test = y_bal[:split], y_bal[split:]

train_dl = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=BATCH, shuffle=True
)
test_dl = DataLoader(
    TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
    batch_size=BATCH
)

print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

# ── 7. LSTM Model (same architecture as lstm_predictor.py) ───────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden=32):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.dense   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return torch.sigmoid(self.dense(out))

model     = LSTMModel()
# Weighted BCE to handle remaining imbalance
pos_weight = torch.tensor([len(normal_idx) / max(len(fail_idx), 1)]).clamp(1, 50)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
print(f"pos_weight: {pos_weight.item():.1f}")

# ── 8. Training ───────────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")
print("-" * 40)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        # BCEWithLogitsLoss needs raw logits, not sigmoid
        out, _ = model.lstm(xb)
        out    = model.dropout(out[:, -1, :])
        logits = model.dense(out).squeeze()
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={total_loss/len(train_dl):.4f}")

# ── 9. Evaluation ─────────────────────────────────────────────────────────────
print("\nEvaluating...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_dl:
        prob = model(xb).squeeze()
        all_preds.extend(prob.numpy())
        all_labels.extend(yb.numpy())

y_pred_prob = np.array(all_preds)
y_true      = np.array(all_labels)
y_pred_bin  = (y_pred_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_true.astype(int), y_pred_bin)
mae      = np.mean(np.abs(y_pred_prob - y_true))
rmse     = np.sqrt(np.mean((y_pred_prob - y_true) ** 2))

print("\n" + "=" * 50)
print("       LSTM Backblaze — Evaluation Results")
print("=" * 50)
print(f"  Accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"  MAE      : {mae:.4f}")
print(f"  RMSE     : {rmse:.4f}")
print()
print(classification_report(y_true.astype(int), y_pred_bin,
                             target_names=["Normal", "Failure"], zero_division=0))
print("=" * 50)

# ── 10. Save model ────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
torch.save(model.state_dict(), MODEL_OUT)
print(f"\nModel saved to: {MODEL_OUT}")
print("\nThis model was trained on REAL Backblaze Q1 2020 data")
print(f"  - {len(csv_files)} days of disk health data")
print(f"  - {disks_used:,} unique disks")
print(f"  - {len(X):,} time-series sequences")
print(f"  - {int(y.sum()):,} real failure events")
print("\nFeature mapping:")
print("  read_errors  ← SMART 5  (Reallocated Sectors Count)")
print("  write_errors ← SMART 187 (Reported Uncorrectable Errors)")
print("  temperature  ← SMART 194 (Temperature Celsius)")
print("  health_score ← SMART 197+198 (Pending/Uncorrectable Sectors)")
