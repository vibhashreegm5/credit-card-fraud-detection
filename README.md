# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning classification models. Credit card fraud detection is an important business and risk analytics problem because fraudulent transactions are rare but can cause serious financial losses. The dataset used in this project is highly imbalanced, meaning that normal transactions are much higher than fraud transactions.

The main aim of this project is to build and compare machine learning models that can correctly identify fraudulent transactions while reducing false positives and false negatives.

---

## Problem Statement

Financial institutions process thousands of transactions every day. Among these transactions, only a very small percentage are fraudulent. Because of this imbalance, a model can achieve very high accuracy by simply predicting most transactions as non-fraud. Therefore, accuracy alone is not a reliable measure for this project.

This project focuses on the following evaluation metrics:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

These metrics are more useful for fraud detection because they show how well the model detects actual fraud cases.

---

## Dataset

The dataset used is the **Credit Card Fraud Detection Dataset** from Kaggle.

Dataset link:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains anonymized credit card transaction records. Most features are transformed using PCA for privacy reasons.

### Target Variable

The target variable is `Class`.

| Class | Meaning |
|---|---|
| 0 | Non-Fraud Transaction |
| 1 | Fraud Transaction |

---

## Tools and Technologies Used

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- SMOTE
- Logistic Regression
- Random Forest Classifier
- Jupyter Notebook

---

## Project Workflow

The project follows a complete machine learning workflow:

1. Import required libraries
2. Load the dataset
3. Understand the dataset structure
4. Check missing values
5. Explore class imbalance
6. Perform exploratory data analysis
7. Scale `Time` and `Amount` variables
8. Split the data into training and testing sets
9. Apply SMOTE on the training data
10. Train Logistic Regression model
11. Train Random Forest model
12. Evaluate model performance
13. Compare model results
14. Analyze feature importance
15. Save model results

---

## Exploratory Data Analysis

The target variable showed a strong class imbalance. Most transactions were non-fraud transactions, while fraud transactions were very rare.

The correlation analysis showed that some variables had a stronger relationship with the fraud class compared to others.

### Important Correlations with Fraud Class

The variables most positively correlated with fraud were:

| Feature | Correlation with Class |
|---|---:|
| V11 | 0.1549 |
| V4 | 0.1334 |
| V2 | 0.0913 |
| V21 | 0.0404 |
| V19 | 0.0348 |

The variables most negatively correlated with fraud were:

| Feature | Correlation with Class |
|---|---:|
| V17 | -0.3265 |
| V14 | -0.3025 |
| V12 | -0.2606 |
| V10 | -0.2169 |
| V16 | -0.1965 |

This suggests that variables such as `V17`, `V14`, `V12`, `V10`, and `V16` are important indicators in identifying fraudulent transactions.

---

## Handling Class Imbalance

Since fraud cases are very rare, the dataset is highly imbalanced. To address this issue, SMOTE was applied only on the training data after train-test split.

This step is important because applying SMOTE before splitting the data can cause data leakage.

### Training Class Distribution After SMOTE

| Class | Count |
|---|---:|
| 0 | 227,451 |
| 1 | 227,451 |

After applying SMOTE, both fraud and non-fraud classes in the training data became balanced.

---

## Machine Learning Models Used

Two machine learning models were trained and compared:

### 1. Logistic Regression with SMOTE

Logistic Regression was trained using the SMOTE-balanced training dataset. This model performed very well in identifying fraud cases but produced many false positives.

### 2. Random Forest Classifier

Random Forest was trained on the original training data with balanced class weights. This model performed better overall, especially in terms of precision and F1-score.

---

## Model Evaluation

### Logistic Regression with SMOTE

| Metric | Value |
|---|---:|
| Accuracy | 0.9743 |
| Precision | 0.0581 |
| Recall | 0.9184 |
| F1-score | 0.1094 |
| ROC-AUC | 0.9698 |

### Confusion Matrix: Logistic Regression with SMOTE

| Actual / Predicted | Non-Fraud | Fraud |
|---|---:|---:|
| Non-Fraud | 55,406 | 1,458 |
| Fraud | 8 | 90 |

Logistic Regression achieved very high recall of 0.9184, meaning it detected most fraud cases. However, its precision was very low at 0.0581, meaning many normal transactions were incorrectly classified as fraud.

---

### Random Forest

| Metric | Value |
|---|---:|
| Accuracy | 0.9995 |
| Precision | 0.9610 |
| Recall | 0.7551 |
| F1-score | 0.8457 |
| ROC-AUC | 0.9581 |

### Confusion Matrix: Random Forest

| Actual / Predicted | Non-Fraud | Fraud |
|---|---:|---:|
| Non-Fraud | 56,861 | 3 |
| Fraud | 24 | 74 |

Random Forest achieved very high precision of 0.9610, meaning most transactions predicted as fraud were actually fraud. However, its recall was 0.7551, meaning it missed 24 fraud cases.

---

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression with SMOTE | 0.9743 | 0.0581 | 0.9184 | 0.1094 | 0.9698 |
| Random Forest | 0.9995 | 0.9610 | 0.7551 | 0.8457 | 0.9581 |

---

## Results Interpretation

The Logistic Regression model with SMOTE achieved the highest recall, identifying 90 out of 98 fraud transactions. This makes it useful when the main goal is to capture as many fraud cases as possible. However, it incorrectly classified 1,458 normal transactions as fraud, which may create unnecessary investigation costs.

The Random Forest model achieved a much better balance between precision and recall. It correctly identified 74 fraud transactions and made only 3 false fraud predictions. Although it missed 24 fraud cases, it achieved a much higher F1-score of 0.8457 compared to Logistic Regression.

Overall, Random Forest is selected as the better model because it provides stronger precision, F1-score, and practical fraud detection performance.

---

## Feature Importance

The Random Forest model identified the following top 10 important features:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | V14 | 0.2099 |
| 2 | V10 | 0.1174 |
| 3 | V4 | 0.1144 |
| 4 | V17 | 0.0873 |
| 5 | V12 | 0.0726 |
| 6 | V11 | 0.0718 |
| 7 | V3 | 0.0689 |
| 8 | V16 | 0.0374 |
| 9 | V7 | 0.0312 |
| 10 | V2 | 0.0254 |

The most important feature was `V14`, followed by `V10`, `V4`, `V17`, and `V12`. These variables contributed the most to the Random Forest model's fraud detection performance.

---

## Final Model Selection

The final selected model is:

### Random Forest Classifier

Random Forest was selected because it achieved:

- Very high accuracy
- Very high precision
- Strong F1-score
- Very low false positives
- Useful feature importance output

Although Logistic Regression had higher recall, Random Forest is more practical for fraud detection because it reduces unnecessary false fraud alerts.

---

## Key Findings

- The dataset is highly imbalanced, with fraud transactions being very rare.
- SMOTE helped balance the training data for Logistic Regression.
- Logistic Regression detected more fraud cases but produced many false positives.
- Random Forest produced much fewer false positives and had a stronger F1-score.
- Features such as `V14`, `V10`, `V4`, `V17`, and `V12` were important for fraud detection.
- Accuracy alone is not enough for fraud detection because of class imbalance.
- Precision, recall, F1-score, and ROC-AUC give a better understanding of model performance.

---

## Business Interpretation

For a financial institution, fraud detection models must balance two risks. The first risk is missing actual fraud transactions. The second risk is incorrectly flagging genuine transactions as fraud. Logistic Regression is better at detecting more fraud cases, but it produces too many false alerts. Random Forest gives a more balanced result by identifying fraud transactions with high precision and fewer false positives.

Therefore, Random Forest is more suitable for practical fraud detection where investigation cost and customer inconvenience should also be considered.

---

## Limitations

This project has some limitations:

- Most features are PCA-transformed, so the original meaning of variables is unknown.
- The dataset is anonymized, which limits business-level interpretation.
- The dataset contains historical transactions only.
- The model was not deployed in a real-time fraud detection system.
- Hyperparameter tuning was not deeply performed in the basic version.

---

## Future Improvements

Future improvements can include:

- Applying XGBoost or LightGBM
- Performing hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Using threshold tuning to improve fraud recall
- Comparing SMOTE with undersampling methods
- Building a Streamlit web application
- Creating a Power BI or Tableau fraud monitoring dashboard
- Saving the final model using pickle or joblib
- Testing the model on new unseen transaction data

---

## Author

**Vibhashree GM**  
Data Scientist | Data Analyst | Risk Analyst | Business Analyst
