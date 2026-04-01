"""
Show LSTM Backblaze training results with plots.
Run: python show_lstm_results.py
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Same config as training ───────────────────────────────────────────────────
DATA_DIR  = "C:/Users/kefii/Downloads/data_Q1_2020/"
MODEL_PATH = "ml-service/models/lstm_model.pth"
SEQ_LEN   = 5
MAX_FILES = 7
MAX_DISKS = 5000
FEAT_MAX  = np.array([100.0, 80.0, 80.0, 1.0], dtype=np.float32)
BB_COLS   = ["serial_number","failure","smart_5_raw","smart_187_raw",
             "smart_194_raw","smart_190_raw","smart_197_raw","smart_198_raw"]

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

# ── 1. Reload data ────────────────────────────────────────────────────────────
print("Loading Backblaze data...")
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))[:MAX_FILES]
frames = [pd.read_csv(f, usecols=BB_COLS) for f in csv_files]
raw = pd.concat(frames, ignore_index=True).fillna(0)

raw["read_errors"]  = raw["smart_5_raw"].clip(0, 100)
raw["write_errors"] = raw["smart_187_raw"].clip(0, 80)
raw["temperature"]  = np.where(raw["smart_194_raw"] > 0,
                                raw["smart_194_raw"], raw["smart_190_raw"]).clip(0, 80)
raw["health_score"] = np.where(raw["smart_198_raw"] > 0, 1.0,
                        np.where(raw["smart_197_raw"] > 0, 0.5, 0.0)).astype(np.float32)

FEATURES = ["read_errors", "write_errors", "temperature", "health_score"]

np.random.seed(42)
all_serials = raw["serial_number"].unique()
if len(all_serials) > MAX_DISKS:
    all_serials = np.random.choice(all_serials, MAX_DISKS, replace=False)
    raw = raw[raw["serial_number"].isin(all_serials)]

X_list, y_list = [], []
for _, group in raw.groupby("serial_number"):
    vals   = group[FEATURES].values.astype(np.float32)
    labels = group["failure"].values.astype(np.float32)
    if len(vals) < SEQ_LEN + 1:
        continue
    for i in range(len(vals) - SEQ_LEN):
        X_list.append(vals[i:i+SEQ_LEN])
        y_list.append(labels[i+SEQ_LEN])

X = np.clip(np.array(X_list, dtype=np.float32) / FEAT_MAX, 0, 1)
y = np.array(y_list, dtype=np.float32)

# Balance
fail_idx   = np.where(y == 1)[0]
normal_idx = np.where(y == 0)[0]
target     = max(len(fail_idx), len(normal_idx) // 10)
if len(fail_idx) > 0 and len(fail_idx) < target:
    repeat = int(np.ceil(target / len(fail_idx)))
    fail_over = np.tile(fail_idx, repeat)[:target]
    all_idx = np.concatenate([normal_idx, fail_over])
else:
    all_idx = np.arange(len(X))
np.random.shuffle(all_idx)
X_bal, y_bal = X[all_idx], y[all_idx]

split = int(len(X_bal) * 0.8)
X_test = X_bal[split:]
y_test = y_bal[split:]

# ── 2. Load model & evaluate ──────────────────────────────────────────────────
print("Loading model and evaluating...")
model = LSTMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

test_dl = DataLoader(
    TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
    batch_size=256
)

all_probs, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        prob = model(xb).squeeze()
        all_probs.extend(prob.numpy())
        all_labels.extend(yb.numpy())

y_prob = np.array(all_probs)
y_true = np.array(all_labels).astype(int)
y_pred = (y_prob >= 0.5).astype(int)

acc  = accuracy_score(y_true, y_pred)
cm   = confusion_matrix(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

print(f"\nAccuracy : {acc:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")
print(classification_report(y_true, y_pred,
      target_names=["Normal", "Failure"], zero_division=0))

# ── 3. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("LSTM Model — Backblaze Q1 2020 Results", fontsize=16, fontweight='bold')

# ── Plot 1: Confusion Matrix ──────────────────────────────────────────────────
ax = axes[0, 0]
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_title("Confusion Matrix", fontsize=13, fontweight='bold')
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Normal", "Failure"])
ax.set_yticklabels(["Normal", "Failure"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=14, color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax)

# ── Plot 2: ROC Curve ─────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC AUC = {roc_auc:.4f}')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random classifier')
ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)

# ── Plot 3: Prediction probability distribution ───────────────────────────────
ax = axes[1, 0]
normal_probs  = y_prob[y_true == 0]
failure_probs = y_prob[y_true == 1]
ax.hist(normal_probs,  bins=40, alpha=0.7, color='#4CAF50', label='Normal disks')
ax.hist(failure_probs, bins=40, alpha=0.7, color='#F44336', label='Failing disks')
ax.axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Threshold (0.5)')
ax.set_xlabel("Predicted Failure Probability")
ax.set_ylabel("Number of Sequences")
ax.set_title("Prediction Distribution", fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# ── Plot 4: Summary metrics bar chart ────────────────────────────────────────
ax = axes[1, 1]
from sklearn.metrics import precision_score, recall_score, f1_score
metrics = {
    'Accuracy':  acc,
    'Precision\n(Failure)': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
    'Recall\n(Failure)':    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    'F1-Score\n(Failure)':  f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    'ROC AUC':  roc_auc,
}
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
bars = ax.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.85, edgecolor='white')
ax.set_ylim(0, 1.15)
ax.set_title("Performance Metrics", fontsize=13, fontweight='bold')
ax.set_ylabel("Score")
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
out_path = "lstm_backblaze_results.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {out_path}")
plt.show()
