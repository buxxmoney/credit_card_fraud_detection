# Credit Card Fraud Detection

## Overview
This project aims to detect fraudulent credit card transactions using machine learning models. The goal is to develop a predictive model that accurately identifies fraudulent transactions while minimizing false alarms, enhancing trust in digital payment systems.

## Project Structure
The repository consists of the following files:

- **`Project.ipynb`**: A Jupyter Notebook containing the code for data preprocessing, feature engineering, model training, evaluation, and visualization. It includes step-by-step documentation and inline comments to facilitate understanding of the model development process.

- **`Data_Mining_Final_Project-3.pdf`**: A comprehensive project report detailing the problem, dataset, feature engineering, model selection, evaluation metrics, and conclusions. This document serves as an in-depth guide to the project, complete with charts, graphs, and performance results.

---

## Problem Statement
Credit card fraud detection is critical for financial security. In 2023, global fraud losses reached $10 billion. This project aims to develop a machine learning model to accurately identify fraudulent transactions while maintaining low false positive rates. The task is framed as a binary classification problem, predicting whether a transaction is fraudulent (`1`) or legitimate (`0`).

---

## Dataset
The dataset was sourced from Kaggle's **Credit Card Fraud Detection Dataset**, which contains anonymized data from European cardholders. Key points include:

- **Number of Transactions**: 284,807
- **Non-Fraudulent Transactions**: 284,315 (99.83%)
- **Fraudulent Transactions**: 492 (0.17%)
- **Features**:
  - 28 anonymized PCA-transformed features (`V1` to `V28`)
  - `Time`: The time elapsed since the first transaction
  - `Amount`: The transaction amount

---

## Methodology

### **1. Data Preprocessing**
- **Handling Imbalance**: The extreme class imbalance (99.83% legitimate vs. 0.17% fraud) was addressed using the **Synthetic Minority Oversampling Technique (SMOTE)**.
- **Feature Scaling**: The `Amount` and `Time` features were scaled to match the range of other features.
- **Data Split**: 
  - Training/Validation: 80% of the data
  - Test: 20% of the data 
  - The training set was further split into 75% for training and 25% for validation.

---

## Algorithms and Models

Three machine learning models were implemented and compared for performance.

### **1. Logistic Regression**
- **Tuning**: 
  - `class_weight='balanced'` to handle class imbalance
  - Regularization strength `C` to prevent overfitting
- **Performance**: High recall (0.90) but low precision (0.06), resulting in a poor F1-score (0.11).

### **2. Random Forest (Best Performing Model)**
- **Tuning**:
  - `n_estimators`: Number of trees
  - `class_weight='balanced'`: Prioritizes the minority class
  - `max_depth`: Controls tree depth to prevent overfitting
  - `min_samples_leaf`: Ensures leaf nodes have enough samples
- **Performance**: 
  - **F1-Score**: 0.83
  - **Precision**: 0.84
  - **Recall**: 0.83
  - **ROC-AUC**: 0.971
  - Detected 81 out of 98 fraudulent cases with only 16 false positives.

### **3. XGBoost**
- **Tuning**:
  - `learning_rate`: Step size for each iteration
  - `n_estimators`: Number of iterations
  - `max_depth`: Limits tree depth to avoid overfitting
  - `scale_pos_weight`: Handles class imbalance by increasing the weight of the minority class
  - `gamma`: Controls minimum loss reduction for splits
- **Performance**: Comparable to Random Forest, but Random Forest had a higher F1-score and was less sensitive to overfitting.

---

## Evaluation and Results

The models were evaluated using the following metrics:
- **F1-Score**: Measures the balance between precision and recall.
- **Precision**: The proportion of correctly predicted fraudulent transactions out of all predicted fraudulent transactions.
- **Recall**: The proportion of fraudulent transactions correctly identified out of all actual fraudulent transactions.
- **ROC-AUC**: Measures the model's ability to distinguish between classes.

| **Metric**         | **Logistic Regression** | **Random Forest** | **XGBoost** |
|-------------------|------------------------|-------------------|-------------|
| **Precision**      | 0.06                   | 0.84               | 0.81        |
| **Recall**         | 0.90                   | 0.83               | 0.80        |
| **F1-Score**       | 0.11                   | 0.83               | 0.82        |
| **ROC-AUC**        | N/A                    | **0.971**          | N/A         |

The **Random Forest** model outperformed Logistic Regression and XGBoost, demonstrating high precision, recall, and ROC-AUC. It achieved a balance between identifying fraud cases and minimizing false alarms.

---

## Key Insights
- **SMOTE** effectively balanced the class distribution, improving model performance.
- **Random Forest** provided the best balance between precision and recall, making it ideal for real-world fraud detection scenarios.
- **Hyperparameter Tuning** for each model was critical in optimizing performance.

---

## How to Run
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
