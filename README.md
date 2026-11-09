# Data Analysis of Signals Data

## Overview
This repository contains an end-to-end data science and machine learning pipeline for analyzing semiconductor manufacturing sensor/signal data (SECOM dataset). The objective is to monitor signal features, detect anomalous manufacturing behaviors, and predict pass/fail test results for quality control.

---

## Project Structure
```
├── NMvijeyendranath_edufyi_capstone2_project.ipynb  # Interactive Jupyter Notebook for exploration & modeling
├── preprocess_and_export.py                         # Data cleaning, missing value imputation & SMOTE pipeline
├── signal_analysis.py                               # Comprehensive EDA, feature selection, PCA, & traditional ML models
├── signal_analysis_gpu.py                           # Deep learning classification pipeline with PyTorch (GPU accelerated)
├── sensor-data.csv                                  # Raw sensor signals dataset
├── preprocessed_data/                               # Processed, balanced, and train/test partitioned CSV datasets
└── outputs/                                         # Generated figures, plots, and saved model artifacts (.pkl, .pt)
```

---

## Key Workflow Steps

1. **Exploratory Data Analysis (EDA):**
   - Signal distributions, univariate/bivariate checks, and feature variance profiling.
   - Target imbalance analysis (Pass vs. Fail classification).
   - Correlation heatmaps and feature-target relationship inspection.

2. **Data Cleansing & Preprocessing:**
   - Removal of constant and high-null signal features.
   - Missing value imputation using median strategy.
   - Class balancing via Synthetic Minority Over-sampling Technique (SMOTE).
   - Train/test splitting with standard scaling.

3. **Dimensionality Reduction & Feature Selection:**
   - Variance thresholding.
   - Principal Component Analysis (PCA) 2D visualization and cumulative explained variance.

4. **Model Training & Benchmarking:**
   - Traditional ML: Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine (SVM), and Gaussian Naive Bayes.
   - Deep Learning: PyTorch Neural Network architecture with GPU acceleration.
   - Hyperparameter optimization, stratified k-fold cross-validation, confusion matrices, and ROC-AUC curve comparisons.

---

## Getting Started

### Prerequisites
- Python 3.9+
- Recommended packages:
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn torch joblib
```

### Running the Analysis
- **Data Cleansing & Preprocessing:**
  ```bash
  python preprocess_and_export.py
  ```
- **Full Signal Analysis & Traditional ML:**
  ```bash
  python signal_analysis.py
  ```
- **GPU Deep Learning Pipeline:**
  ```bash
  python signal_analysis_gpu.py
  ```
