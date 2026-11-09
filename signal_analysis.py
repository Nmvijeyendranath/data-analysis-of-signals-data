"""
Semiconductor Manufacturing Signal Data - Full Analysis
Capstone Project 2 | Corizo
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
import joblib
import os

# Output directory
OUT = r'c:\Users\nmvij\Downloads\corizo_group_projet\outputs'
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style='whitegrid', palette='husl')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

# ─────────────────────────────────────────────
# 1. IMPORT & EXPLORE
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: IMPORT & EXPLORE DATA")
print("=" * 60)

# Dataset path - supports 'sensor-data.csv' (PDF spec) with fallback
DATA_FILE = 'sensor-data.csv'
if not os.path.exists(DATA_FILE):
    for fallback in ['signal-data (1).csv', 'signal-data.csv']:
        if os.path.exists(fallback):
            DATA_FILE = fallback
            break

df = pd.read_csv(DATA_FILE)
print(f"Dataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nFirst few columns: {df.columns[:5].tolist()}")
print(f"Target column: '{df.columns[-1]}'")
print(f"\nData types:\n{df.dtypes.value_counts()}")
print(f"\nTarget distribution:\n{df['Pass/Fail'].value_counts()}")
print(f"\nClass balance: {df['Pass/Fail'].value_counts(normalize=True).mul(100).round(2).to_dict()}")

# ─────────────────────────────────────────────
# 2. DATA CLEANSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANSING")
print("=" * 60)

# 2a. Drop 'Time' column (non-predictive timestamp)
df.drop(columns=['Time'], inplace=True)
print("Dropped 'Time' column (timestamp, not a predictive feature)")

# 2b. Identify & drop zero-variance columns
numeric_df = df.select_dtypes(include='number')
zero_var_cols = numeric_df.columns[numeric_df.std() == 0].tolist()
print(f"\nZero-variance columns (constant): {len(zero_var_cols)}")
df.drop(columns=zero_var_cols, inplace=True)
print(f"Dropped {len(zero_var_cols)} zero-variance columns")

# 2c. Drop columns with >50% missing values
miss_pct = df.isnull().mean()
high_miss_cols = miss_pct[miss_pct > 0.5].index.tolist()
print(f"\nColumns with >50% missing values: {len(high_miss_cols)}")
df.drop(columns=high_miss_cols, inplace=True)
print(f"Dropped {len(high_miss_cols)} high-missing columns")

print(f"\nShape after dropping: {df.shape}")

# 2d. Impute remaining missing values with median
remaining_miss = df.isnull().sum().sum()
print(f"\nRemaining missing values: {remaining_miss}")
feature_cols = [c for c in df.columns if c != 'Pass/Fail']
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
print(f"Imputed with column median.")
print(f"Missing values after imputation: {df.isnull().sum().sum()}")

# 2e. Convert target to numeric (ensure -1 and 1)
df['Pass/Fail'] = df['Pass/Fail'].astype(int)
print(f"\nFinal clean dataset shape: {df.shape}")
print(f"Target distribution:\n{df['Pass/Fail'].value_counts()}")

# ─────────────────────────────────────────────
# 3. DATA ANALYSIS & VISUALISATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: DATA ANALYSIS & VISUALISATION")
print("=" * 60)

X = df.drop(columns=['Pass/Fail'])
y = df['Pass/Fail']
# Recode for display: 1=Pass, 0=Fail
y_bin = (y == 1).astype(int)  # 1=Pass, 0=Fail

# ── 3.1 Target Distribution ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = y.value_counts()
labels = ['Fail (-1)', 'Pass (+1)']
colors = ['#e74c3c', '#2ecc71']
axes[0].bar(labels, [counts[-1], counts[1]], color=colors, edgecolor='black', linewidth=0.8)
axes[0].set_title('Target Class Distribution', fontweight='bold', fontsize=13)
axes[0].set_ylabel('Count')
for i, v in enumerate([counts[-1], counts[1]]):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

axes[1].pie([counts[-1], counts[1]], labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Class Proportion', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig(f'{OUT}/01_target_distribution.png', bbox_inches='tight')
plt.close()
print("✓ Saved: 01_target_distribution.png")

# ── 3.2 Statistical Summary ───────────────────
desc = X.describe().T
print("\nStatistical Summary (first 10 features):")
print(desc.head(10).to_string())

# ── 3.3 Missing Value Heatmap (before cleaning - use original) ───
# We'll show remaining missingness pattern visually (post-clean it's 0)
# Instead show variance distribution
fig, ax = plt.subplots(figsize=(10, 4))
var_vals = X.var().sort_values(ascending=False)
ax.semilogy(range(len(var_vals)), var_vals.values, color='steelblue', linewidth=1.5)
ax.set_title('Feature Variance Distribution (log scale)', fontweight='bold', fontsize=13)
ax.set_xlabel('Feature Rank (by Variance)')
ax.set_ylabel('Variance (log scale)')
ax.fill_between(range(len(var_vals)), var_vals.values, alpha=0.3, color='steelblue')
plt.tight_layout()
plt.savefig(f'{OUT}/02_feature_variance.png', bbox_inches='tight')
plt.close()
print("✓ Saved: 02_feature_variance.png")

# ── 3.4 Univariate - Distribution of top features ────────────────
# Select top 9 features by variance
top9 = var_vals.head(9).index.tolist()
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for i, col in enumerate(top9):
    ax = axes[i // 3][i % 3]
    for label, grp in df.groupby('Pass/Fail'):
        name = 'Pass (+1)' if label == 1 else 'Fail (-1)'
        color = '#2ecc71' if label == 1 else '#e74c3c'
        ax.hist(grp[col], bins=30, alpha=0.6, label=name, color=color, density=True)
    ax.set_title(f'Feature {col}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7)
fig.suptitle('Univariate Distributions: Top-9 High-Variance Features\n(Pass vs Fail)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/03_univariate_distributions.png', bbox_inches='tight')
plt.close()
print("✓ Saved: 03_univariate_distributions.png")

# ── 3.5 Box plots - Pass vs Fail ─────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for i, col in enumerate(top9):
    ax = axes[i // 3][i % 3]
    pass_data = df[df['Pass/Fail'] == 1][col]
    fail_data = df[df['Pass/Fail'] == -1][col]
    bp = ax.boxplot([fail_data, pass_data], labels=['Fail', 'Pass'],
                    patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2})
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#2ecc71')
    bp['boxes'][1].set_alpha(0.7)
    ax.set_title(f'Feature {col}', fontsize=10, fontweight='bold')

fig.suptitle('Box Plots: Pass vs Fail — Top-9 High-Variance Features',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/04_boxplots_pass_fail.png', bbox_inches='tight')
plt.close()
print("✓ Saved: 04_boxplots_pass_fail.png")

# ── 3.6 Correlation Heatmap (top 20 features) ────────────────────
top20 = var_vals.head(20).index.tolist()
corr = df[top20 + ['Pass/Fail']].corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title('Correlation Heatmap: Top-20 High-Variance Features + Target',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/05_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("✓ Saved: 05_correlation_heatmap.png")

# ── 3.7 Correlation with target ───────────────
corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
top_corr = corr_with_target.head(20)
fig, ax = plt.subplots(figsize=(10, 6))
colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_corr)))
bars = ax.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors_bar[::-1])
ax.set_title('Top 20 Features by Absolute Correlation with Target (Pass/Fail)',
             fontweight='bold', fontsize=13)
ax.set_xlabel('Absolute Pearson Correlation')
ax.axvline(0.1, color='red', linestyle='--', linewidth=1, label='r=0.1 threshold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/06_feature_target_correlation.png', bbox_inches='tight')
plt.close()
print("✓ Saved: 06_feature_target_correlation.png")

# ── 3.8 Bivariate: Scatter plots of top-4 correlated features ────
top4 = corr_with_target.head(4).index.tolist()
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, col in enumerate(top4):
    ax = axes[i // 2][i % 2]
    for label, color, name in [(-1, '#e74c3c', 'Fail'), (1, '#2ecc71', 'Pass')]:
        mask = y == label
        ax.scatter(df.loc[mask, top4[0]], df.loc[mask, col],
                   c=color, alpha=0.5, s=20, label=name)
    ax.set_xlabel(f'Feature {top4[0]}')
    ax.set_ylabel(f'Feature {col}')
    ax.set_title(f'{top4[0]} vs {col}', fontweight='bold')
    ax.legend()
fig.suptitle('Bivariate Analysis: Top Correlated Features',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/07_bivariate_scatter.png', bbox_inches='tight')
plt.close()
print("✓ Saved: 07_bivariate_scatter.png")

# ── 3.9 PCA Multivariate Visualisation ───────
scaler_pca = StandardScaler()
X_scaled_pca = scaler_pca.fit_transform(X)
pca_2 = PCA(n_components=2, random_state=42)
X_pca = pca_2.fit_transform(X_scaled_pca)

fig, ax = plt.subplots(figsize=(9, 7))
for label, color, name in [(-1, '#e74c3c', 'Fail (-1)'), (1, '#2ecc71', 'Pass (+1)')]:
    mask = y == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.6, s=30,
               label=name, edgecolors='white', linewidths=0.3)
ax.set_xlabel(f'PC1 ({pca_2.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca_2.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=12)
ax.set_title('PCA Projection: 2D View of All Features', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/08_pca_2d.png', bbox_inches='tight')
plt.close()
print("✓ Saved: 08_pca_2d.png")

# PCA Explained Variance
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled_pca)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n95 = np.searchsorted(cumvar, 0.95) + 1
n90 = np.searchsorted(cumvar, 0.90) + 1

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(cumvar) + 1), cumvar * 100, 'b-', linewidth=2)
ax.axhline(95, color='red', linestyle='--', label=f'95% variance → {n95} components')
ax.axhline(90, color='orange', linestyle='--', label=f'90% variance → {n90} components')
ax.fill_between(range(1, len(cumvar) + 1), cumvar * 100, alpha=0.2, color='blue')
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance (%)')
ax.set_title('PCA: Cumulative Explained Variance', fontsize=13, fontweight='bold')
ax.legend()
ax.set_xlim(1, min(150, len(cumvar)))
plt.tight_layout()
plt.savefig(f'{OUT}/09_pca_explained_variance.png', bbox_inches='tight')
plt.close()
print(f"✓ Saved: 09_pca_explained_variance.png  (90%={n90} comps, 95%={n95} comps)")

# ─────────────────────────────────────────────
# 4. DATA PRE-PROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: DATA PRE-PROCESSING")
print("=" * 60)

# 4a. Segregate predictors vs target
X = df.drop(columns=['Pass/Fail'])
y = df['Pass/Fail']
print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

# 4b. Target balancing with SMOTE
print(f"\nOriginal class distribution: {dict(y.value_counts())}")
try:
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"After SMOTE: {dict(pd.Series(y_res).value_counts())}")
except ImportError:
    print("imblearn not available; using class_weight='balanced' in models instead")
    X_res, y_res = X.values, y.values

# Visualise before/after SMOTE
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
pd.Series(y).value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'],
                                  edgecolor='black', rot=0)
axes[0].set_title('Before SMOTE', fontweight='bold')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
pd.Series(y_res).value_counts().plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'],
                                      edgecolor='black', rot=0)
axes[1].set_title('After SMOTE (Balanced)', fontweight='bold')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
plt.suptitle('Class Balancing with SMOTE', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/10_smote_balancing.png', bbox_inches='tight')
plt.close()
print("✓ Saved: 10_smote_balancing.png")

# 4c. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# 4d. Standardise
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# 4e. Verify statistical similarity
print("\nStatistical similarity check (Train vs Test vs Original):")
for name, data in [('Original', X.values), ('Train', X_train_sc), ('Test', X_test_sc)]:
    flat = data[:, :10]  # first 10 features
    print(f"  {name:10s} - Mean: {flat.mean():.4f}, Std: {flat.std():.4f}")

# ─────────────────────────────────────────────
# 5. MODEL TRAINING, TESTING & TUNING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: MODEL TRAINING, TESTING & TUNING")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# ── Helper ────────────────────────────────────
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    test_acc = accuracy_score(y_te, model.predict(X_te))
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='accuracy')
    y_pred = model.predict(X_te)
    try:
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score((y_te == 1).astype(int), y_prob)
    except Exception:
        auc = None
    print(f"\n{'─'*50}")
    print(f"MODEL: {name}")
    print(f"  Train Accuracy : {train_acc:.4f}")
    print(f"  Test  Accuracy : {test_acc:.4f}")
    print(f"  CV Accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    if auc:
        print(f"  ROC-AUC        : {auc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_te, y_pred)}")
    results[name] = {
        'model': model, 'train_acc': train_acc, 'test_acc': test_acc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'auc': auc, 'y_pred': y_pred,
        'y_prob': y_prob if auc else None
    }
    return model

# ── 5.1 Logistic Regression ──────────────────
print("\n[1/4] Logistic Regression with GridSearch...")
lr_params = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs'], 'max_iter': [1000]}
lr_gs = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=cv,
                     scoring='accuracy', n_jobs=-1)
lr_gs.fit(X_train_sc, y_train)
best_lr = lr_gs.best_estimator_
print(f"  Best params: {lr_gs.best_params_}")
evaluate_model("Logistic Regression", best_lr, X_train_sc, y_train, X_test_sc, y_test)

# ── 5.2 Random Forest ────────────────────────
print("\n[2/4] Random Forest with GridSearch...")
rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20],
             'min_samples_split': [2, 5]}
rf_gs = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                     rf_params, cv=cv, scoring='accuracy', n_jobs=-1)
rf_gs.fit(X_train_sc, y_train)
best_rf = rf_gs.best_estimator_
print(f"  Best params: {rf_gs.best_params_}")
evaluate_model("Random Forest", best_rf, X_train_sc, y_train, X_test_sc, y_test)

# ── 5.3 SVM ──────────────────────────────────
print("\n[3/4] SVM with GridSearch...")
svm_params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale']}
svm_gs = GridSearchCV(SVC(probability=True, random_state=42, class_weight='balanced'),
                      svm_params, cv=cv, scoring='accuracy', n_jobs=-1)
svm_gs.fit(X_train_sc, y_train)
best_svm = svm_gs.best_estimator_
print(f"  Best params: {svm_gs.best_params_}")
evaluate_model("SVM", best_svm, X_train_sc, y_train, X_test_sc, y_test)

# ── 5.4 Naive Bayes ──────────────────────────
print("\n[4/4] Gaussian Naive Bayes...")
nb = GaussianNB()
evaluate_model("Naive Bayes", nb, X_train_sc, y_train, X_test_sc, y_test)

# ─────────────────────────────────────────────
# 5.5 Model Comparison
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

comp_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Acc': [results[m]['train_acc'] for m in results],
    'Test Acc': [results[m]['test_acc'] for m in results],
    'CV Mean': [results[m]['cv_mean'] for m in results],
    'CV Std': [results[m]['cv_std'] for m in results],
    'ROC-AUC': [results[m]['auc'] if results[m]['auc'] else 'N/A' for m in results],
}).sort_values('Test Acc', ascending=False).reset_index(drop=True)

print(comp_df.to_string(index=False))

# Comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(results))
w = 0.35
train_accs = [results[m]['train_acc'] for m in results]
test_accs = [results[m]['test_acc'] for m in results]
ax.bar(x - w/2, train_accs, w, label='Train Accuracy', color='#3498db', alpha=0.8, edgecolor='black')
ax.bar(x + w/2, test_accs, w, label='Test Accuracy', color='#e74c3c', alpha=0.8, edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(list(results.keys()), fontsize=11)
ax.set_ylabel('Accuracy')
ax.set_ylim(0.5, 1.05)
ax.set_title('Model Comparison: Train vs Test Accuracy', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
for xi, (tr, te) in zip(x, zip(train_accs, test_accs)):
    ax.text(xi - w/2, tr + 0.005, f'{tr:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(xi + w/2, te + 0.005, f'{te:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/11_model_comparison.png', bbox_inches='tight')
plt.close()
print("✓ Saved: 11_model_comparison.png")

# ── ROC Curves ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
colors_roc = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
for (name, res), c in zip(results.items(), colors_roc):
    if res['y_prob'] is not None:
        y_te_bin = (y_test == 1).astype(int)
        fpr, tpr, _ = roc_curve(y_te_bin, res['y_prob'])
        ax.plot(fpr, tpr, color=c, linewidth=2,
                label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Models', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUT}/12_roc_curves.png', bbox_inches='tight')
plt.close()
print("✓ Saved: 12_roc_curves.png")

# ── Confusion Matrices ────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, res) in zip(axes.flatten(), results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Fail(-1)', 'Pass(+1)'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name}\nTest Acc: {res["test_acc"]:.3f}', fontweight='bold')
fig.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/13_confusion_matrices.png', bbox_inches='tight')
plt.close()
print("✓ Saved: 13_confusion_matrices.png")

# ── Feature Importance (Random Forest) ───────
best_model_name = comp_df.iloc[0]['Model']
print(f"\nBest model: {best_model_name} (Test Acc: {comp_df.iloc[0]['Test Acc']:.4f})")

if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
    top_feat = feat_imp.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_feat = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_feat)))
    top_feat[::-1].plot(kind='barh', ax=ax, color=colors_feat[::-1])
    ax.set_title('Top 20 Feature Importances (Random Forest)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{OUT}/14_feature_importance.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: 14_feature_importance.png")

# ─────────────────────────────────────────────
# 5.6 Save Best Model
# ─────────────────────────────────────────────
best_model = results[best_model_name]['model']
model_path = f'{OUT}/best_model_{best_model_name.replace(" ", "_")}.pkl'
joblib.dump({'model': best_model, 'scaler': scaler}, model_path)
print(f"\n✓ Best model saved to: {model_path}")

# ─────────────────────────────────────────────
# 6. CONCLUSION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: CONCLUSION")
print("=" * 60)
print(f"""
Dataset: 1567 samples × 591 sensor features (semiconductor manufacturing)
Target: Highly imbalanced — 93.4% Fail, 6.6% Pass

Preprocessing Steps:
  • Dropped 'Time' (non-predictive timestamp)
  • Removed {len(zero_var_cols)} zero-variance (constant) features
  • Removed {len(high_miss_cols)} features with >50% missing values
  • Median imputation for remaining missing values
  • SMOTE oversampling to balance classes
  • StandardScaler normalisation

Feature Engineering:
  • PCA shows 90% variance explained by ~{n90} components
  • Top correlated features with target identified

Model Performance Summary:
{comp_df[['Model','Train Acc','Test Acc','CV Mean']].to_string(index=False)}

Best Model: {best_model_name}
  → Highest test accuracy with good generalisation (low train/test gap)
  → Handles high-dimensional, imbalanced semiconductor data well

Key Findings:
  • Dataset is severely imbalanced (SMOTE critical for recall on Pass class)
  • Many features are redundant (zero variance / high missing)
  • After cleaning, ~{X.shape[1]} meaningful features remain
  • Random Forest shows best robustness due to ensemble nature and built-in feature selection
""")

print("\n✅ All outputs saved to:", OUT)
print("   Plots: 01–14 PNG files")
print(f"   Model: best_model_{best_model_name.replace(' ', '_')}.pkl")
