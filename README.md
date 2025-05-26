# 💳 Credit Card Fraud Detection

A machine learning pipeline for detecting fraudulent credit card transactions using anonymized transaction data. This project tackles the class imbalance problem, evaluates multiple models, and highlights the most effective techniques for real-world fraud detection.

> **Author:** Sebastian Buxman  
> **Tech Stack:** Python, Pandas, Scikit-Learn, Matplotlib, XGBoost

---

## 📌 Objective

To build a supervised learning model that:
- Accurately detects fraudulent transactions
- Minimizes false negatives (fraud that goes undetected)
- Handles extreme class imbalance (fraudulent cases ≪ legitimate ones)

---

## 📊 Dataset

- **Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud Cases:** 492 (≈ 0.17%)
- **Features:** 30 (anonymized PCA components + `Time` and `Amount`)
- **Target:** `Class` (1 = fraud, 0 = legitimate)

---

## ⚙️ Pipeline Overview

1. **Exploratory Data Analysis (EDA)**
   - Distribution analysis
   - Correlation heatmaps
   - Class imbalance visualization

2. **Preprocessing**
   - Feature scaling (`StandardScaler`)
   - Stratified train/test split
   - Resampling using:
     - SMOTE (Synthetic Minority Oversampling)
     - Undersampling

3. **Model Training**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - SVM

4. **Evaluation Metrics**
   - Confusion Matrix
   - Precision / Recall
   - F1-Score
   - ROC-AUC

---

## ✅ Results

| Model              | Precision | Recall | F1-Score | ROC-AUC |
|-------------------|-----------|--------|----------|---------|
| Logistic Regression | 0.93      | 0.67   | 0.78     | 0.97    |
| Random Forest       | 0.94      | 0.82   | 0.88     | 0.98    |
| **XGBoost**         | **0.95**  | **0.87** | **0.91** | **0.99** |
| SVM (RBF)           | 0.88      | 0.70   | 0.78     | 0.96    |

> **XGBoost** was the top-performing model in both ROC-AUC and F1-score.

---

## 📈 Key Insights

- Handling class imbalance was **critical** — naive models performed poorly.
- **SMOTE + XGBoost** yielded the best fraud detection with minimal false negatives.
- Some PCA components strongly influenced predictions even though the raw features were anonymized.

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/buxxmoney/credit_card_fraud_detection
   cd credit_card_fraud_detection
