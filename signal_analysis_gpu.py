"""
Semiconductor Manufacturing Signal Data — GPU-Accelerated Analysis
Capstone Project 2 | Corizo
GPU: NVIDIA GeForce RTX 3050 6GB Laptop | CUDA 12.6 | PyTorch 2.14.1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, time, joblib
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE

# ─── GPU Setup ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 60)
print("GPU SETUP")
print("=" * 60)
print(f"Device        : {device}")
if torch.cuda.is_available():
    print(f"GPU Name      : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version  : {torch.version.cuda}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory    : {total_mem:.2f} GB")
    torch.backends.cudnn.benchmark = True   # auto-tune convolutions
print()

OUT = r'c:\Users\nmvij\Downloads\corizo_group_projet\outputs'
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style='whitegrid', palette='husl')
plt.rcParams['figure.dpi'] = 120

# ─────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN DATA  (same as CPU script)
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1-2: LOAD, CLEAN & PREPROCESS ON GPU")
print("=" * 60)

# Dataset path - supports 'sensor-data.csv' (PDF spec) with fallback
DATA_FILE = 'sensor-data.csv'
if not os.path.exists(DATA_FILE):
    for fallback in ['signal-data (1).csv', 'signal-data.csv']:
        if os.path.exists(fallback):
            DATA_FILE = fallback
            break

df = pd.read_csv(DATA_FILE)
df.drop(columns=['Time'], inplace=True)

# Zero-variance drop
numeric_df = df.select_dtypes(include='number')
zero_var_cols = numeric_df.columns[numeric_df.std() == 0].tolist()
df.drop(columns=zero_var_cols, inplace=True)

# High-missing drop
miss_pct = df.isnull().mean()
df.drop(columns=miss_pct[miss_pct > 0.5].index.tolist(), inplace=True)

# Median imputation
feat_cols = [c for c in df.columns if c != 'Pass/Fail']
df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
df['Pass/Fail'] = df['Pass/Fail'].astype(int)

X_np = df.drop(columns=['Pass/Fail']).values.astype(np.float32)
y_np = df['Pass/Fail'].values.astype(np.int64)
print(f"Clean data shape : {X_np.shape}")
print(f"Classes          : {np.unique(y_np, return_counts=True)}")

# ─────────────────────────────────────────────────────────────────
# GPU PREPROCESSING  —  StandardScaler on GPU via torch
# ─────────────────────────────────────────────────────────────────
t0 = time.time()
X_gpu = torch.tensor(X_np, dtype=torch.float32, device=device)

# Compute mean & std on GPU
mean_gpu = X_gpu.mean(dim=0)
std_gpu  = X_gpu.std(dim=0)
std_gpu[std_gpu == 0] = 1.0          # avoid div-by-zero
X_scaled_gpu = (X_gpu - mean_gpu) / std_gpu

print(f"\nGPU StandardScaler  : {time.time()-t0:.4f}s  (on {device})")
print(f"  mean range : [{mean_gpu.min().item():.4f}, {mean_gpu.max().item():.4f}]")
print(f"  std  range : [{std_gpu.min().item():.4f},  {std_gpu.max().item():.4f}]")

# ─────────────────────────────────────────────────────────────────
# GPU PCA  (torch.linalg.svd)
# ─────────────────────────────────────────────────────────────────
t0 = time.time()
n_comp = 128   # covers ~90% variance (identified in CPU run)
U, S, Vh = torch.linalg.svd(X_scaled_gpu, full_matrices=False)
X_pca_gpu = U[:, :n_comp] * S[:n_comp]          # shape (N, 128)
explained = (S[:n_comp]**2).sum() / (S**2).sum()
print(f"\nGPU PCA ({n_comp} comps)  : {time.time()-t0:.4f}s  — "
      f"explains {explained.item()*100:.1f}% variance")

# Move back to CPU for SMOTE (imblearn doesn't support GPU tensors)
X_pca_np = X_pca_gpu.cpu().numpy()
print(f"\nSMOTE on CPU (RAPIDS not available on Windows)...")
t0 = time.time()
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_pca_np, y_np)
print(f"SMOTE done          : {time.time()-t0:.4f}s  —  shape {X_res.shape}")
print(f"Class balance after : {np.unique(y_res, return_counts=True)}")

# Train / Test split
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42,
    stratify=y_res
)
# Recode -1 → 0 for BCE loss  (+1 → 1, -1 → 0)
y_train_bin = (y_train_np == 1).astype(np.float32)
y_test_bin  = (y_test_np  == 1).astype(np.float32)

# Move to GPU
X_train_gpu = torch.tensor(X_train_np, dtype=torch.float32, device=device)
X_test_gpu  = torch.tensor(X_test_np,  dtype=torch.float32, device=device)
y_train_gpu = torch.tensor(y_train_bin, dtype=torch.float32, device=device).unsqueeze(1)
y_test_gpu  = torch.tensor(y_test_bin,  dtype=torch.float32, device=device).unsqueeze(1)

print(f"\nTrain GPU tensor : {X_train_gpu.shape}  on {X_train_gpu.device}")
print(f"Test  GPU tensor : {X_test_gpu.shape}   on {X_test_gpu.device}")

# ─────────────────────────────────────────────────────────────────
# NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────────────────────────
class SignalNet(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))   # binary output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = SignalNet(
    in_dim=n_comp,
    hidden_dims=[256, 128, 64, 32],
    dropout=0.3
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNeural Network on {device}")
print(f"  Architecture : {n_comp} → 256 → 128 → 64 → 32 → 1")
print(f"  Parameters   : {total_params:,}")
print(model)

# ─────────────────────────────────────────────────────────────────
# TRAINING  —  runs fully on GPU
# ─────────────────────────────────────────────────────────────────
EPOCHS      = 120
BATCH_SIZE  = 128
LR          = 1e-3
WEIGHT_DECAY= 1e-4

# Weighted loss to handle any residual imbalance
pos_weight = torch.tensor(
    [(y_train_bin == 0).sum() / (y_train_bin == 1).sum()],
    device=device
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_ds = TensorDataset(X_train_gpu, y_train_gpu)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

print(f"\nTraining {EPOCHS} epochs | batch={BATCH_SIZE} | lr={LR}")
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Test Acc':>10}  {'LR':>10}")
print("-" * 55)

history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}
t_train_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss, correct, total = 0.0, 0, 0

    for xb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)
        preds = (torch.sigmoid(out) >= 0.5).float()
        correct += (preds == yb).sum().item()
        total   += xb.size(0)

    scheduler.step()

    train_loss = epoch_loss / total
    train_acc  = correct / total

    # Test accuracy (no grad)
    model.eval()
    with torch.no_grad():
        test_out  = model(X_test_gpu)
        test_preds = (torch.sigmoid(test_out) >= 0.5).float()
        test_acc  = (test_preds == y_test_gpu).float().mean().item()

    cur_lr = scheduler.get_last_lr()[0]
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    history['lr'].append(cur_lr)

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>10.4f}  "
              f"{test_acc:>10.4f}  {cur_lr:>10.6f}")

elapsed = time.time() - t_train_start
print(f"\nTraining finished in {elapsed:.2f}s  ({elapsed/EPOCHS*1000:.1f}ms/epoch)")

# ─────────────────────────────────────────────────────────────────
# FINAL EVALUATION
# ─────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits    = model(X_test_gpu)
    probs_gpu = torch.sigmoid(logits).squeeze()
    preds_gpu = (probs_gpu >= 0.5).long()

y_pred_np  = preds_gpu.cpu().numpy()
y_prob_np  = probs_gpu.cpu().numpy()
y_test_bin_np = y_test_bin.astype(int)

# Recode back to -1/+1 for display
y_pred_orig = np.where(y_pred_np == 1, 1, -1)
y_test_orig = np.where(y_test_bin_np == 1, 1, -1)

test_acc  = accuracy_score(y_test_orig, y_pred_orig)
train_acc = history['train_acc'][-1]
auc       = roc_auc_score(y_test_bin_np, y_prob_np)

print("\n" + "=" * 60)
print("NEURAL NETWORK RESULTS (GPU)")
print("=" * 60)
print(f"Train Accuracy  : {train_acc:.4f}")
print(f"Test  Accuracy  : {test_acc:.4f}")
print(f"ROC-AUC         : {auc:.4f}")
print(f"\nClassification Report:\n"
      f"{classification_report(y_test_orig, y_pred_orig)}")

# ─────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────

# -- Training curves --
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(history['train_loss'], color='#e74c3c', linewidth=2)
axes[0].set_title('Training Loss', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')

axes[1].plot(history['train_acc'], label='Train', color='#3498db', linewidth=2)
axes[1].plot(history['test_acc'],  label='Test',  color='#2ecc71', linewidth=2)
axes[1].set_title('Accuracy over Epochs', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].set_ylim(0.5, 1.02)

axes[2].plot(history['lr'], color='#9b59b6', linewidth=2)
axes[2].set_title('Learning Rate Schedule (Cosine)', fontweight='bold')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('LR')

plt.suptitle('GPU Neural Network — Training History\n'
             f'(RTX 3050 | CUDA 12.6 | {elapsed:.1f}s total)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/15_gpu_nn_training_curves.png', bbox_inches='tight')
plt.close()
print("Saved: 15_gpu_nn_training_curves.png")

# -- Confusion matrix --
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test_orig, y_pred_orig)
ConfusionMatrixDisplay(cm, display_labels=['Fail(-1)', 'Pass(+1)']).plot(
    ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'GPU Neural Net\nTest Acc: {test_acc:.4f} | AUC: {auc:.4f}',
             fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/16_gpu_nn_confusion_matrix.png', bbox_inches='tight')
plt.close()
print("Saved: 16_gpu_nn_confusion_matrix.png")

# -- ROC curve --
fpr, tpr, _ = roc_curve(y_test_bin_np, y_prob_np)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color='#e74c3c', linewidth=2.5,
        label=f'GPU Neural Net (AUC = {auc:.4f})')
ax.plot([0,1],[0,1],'k--', linewidth=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — GPU Neural Network', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/17_gpu_nn_roc.png', bbox_inches='tight')
plt.close()
print("Saved: 17_gpu_nn_roc.png")

# -- Full model comparison (CPU models + GPU NN) --
cpu_results = {
    'SVM (CPU)':         {'test_acc': 0.9966, 'auc': 1.0000, 'cv': 0.9953},
    'Random Forest (CPU)': {'test_acc': 0.9863, 'auc': 0.9996, 'cv': 0.9880},
    'Logistic Reg (CPU)':  {'test_acc': 0.9539, 'auc': 0.9563, 'cv': 0.9350},
    'Naive Bayes (CPU)':   {'test_acc': 0.6007, 'auc': 0.6215, 'cv': 0.5919},
    'Neural Net (GPU)':    {'test_acc': test_acc, 'auc': auc,   'cv': None},
}

models_all = list(cpu_results.keys())
test_accs  = [v['test_acc'] for v in cpu_results.values()]
aucs       = [v['auc']      for v in cpu_results.values()]
colors_all = ['#3498db','#2ecc71','#e67e22','#95a5a6','#e74c3c']

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

bars = axes[0].bar(models_all, test_accs, color=colors_all,
                   edgecolor='black', linewidth=0.8)
axes[0].set_ylim(0.5, 1.05)
axes[0].set_ylabel('Test Accuracy', fontsize=11)
axes[0].set_title('Test Accuracy: All Models', fontweight='bold', fontsize=12)
axes[0].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, test_accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].bar_label  # highlight GPU bar
bars[-1].set_edgecolor('gold'); bars[-1].set_linewidth(2.5)

bars2 = axes[1].bar(models_all, aucs, color=colors_all,
                    edgecolor='black', linewidth=0.8)
axes[1].set_ylim(0.5, 1.05)
axes[1].set_ylabel('ROC-AUC', fontsize=11)
axes[1].set_title('ROC-AUC: All Models', fontweight='bold', fontsize=12)
axes[1].tick_params(axis='x', rotation=20)
for bar, val in zip(bars2, aucs):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
bars2[-1].set_edgecolor('gold'); bars2[-1].set_linewidth(2.5)

fig.suptitle('Complete Model Comparison: CPU Models vs GPU Neural Network',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/18_full_model_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: 18_full_model_comparison.png")

# ─────────────────────────────────────────────────────────────────
# SAVE GPU MODEL
# ─────────────────────────────────────────────────────────────────
model_save = {
    'model_state': model.state_dict(),
    'architecture': {'in_dim': n_comp, 'hidden_dims': [256,128,64,32], 'dropout': 0.3},
    'gpu_mean': mean_gpu.cpu().numpy(),
    'gpu_std': std_gpu.cpu().numpy(),
    'pca_Vh': Vh[:n_comp].cpu().numpy(),    # to project new data
    'device': str(device),
    'test_acc': test_acc,
    'auc': auc,
}
torch.save(model_save, f'{OUT}/best_model_GPU_NeuralNet.pt')
print(f"\nGPU model saved: outputs/best_model_GPU_NeuralNet.pt")

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
GPU Used        : {torch.cuda.get_device_name(0)}
CUDA Version    : {torch.version.cuda}
Training Time   : {elapsed:.2f}s for {EPOCHS} epochs

GPU Preprocessing:
  StandardScaler   -> GPU (torch tensors, instant)
  PCA (128 comps)  -> GPU (torch.linalg.svd)
  SMOTE            -> CPU (imblearn, Windows limitation)
  NN Training      -> GPU (fully on CUDA)

Model Results:
  Model                Test Acc   ROC-AUC
  SVM (CPU)            0.9966     1.0000
  Random Forest (CPU)  0.9863     0.9996
  Neural Net (GPU)     {test_acc:.4f}     {auc:.4f}
  Logistic Reg (CPU)   0.9539     0.9563
  Naive Bayes (CPU)    0.6007     0.6215

Outputs saved to: {OUT}
  15_gpu_nn_training_curves.png
  16_gpu_nn_confusion_matrix.png
  17_gpu_nn_roc.png
  18_full_model_comparison.png
  best_model_GPU_NeuralNet.pt
""")
print("Done!")

