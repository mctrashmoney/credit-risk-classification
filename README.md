# Credit Risk Classification - Logistic Regression

## Project Overview
This project builds and evaluates a **Logistic Regression** model to predict whether a loan is **high-risk (1)** or **healthy (0)** based on borrower data. The model is trained using a dataset containing various financial metrics related to loan applicants.

## Dataset
The dataset consists of multiple loan records and includes the following features:
- **loan_size**: Amount of the loan issued
- **interest_rate**: Interest rate associated with the loan
- **borrower_income**: Annual income of the borrower
- **debt_to_income**: Borrower's debt-to-income ratio
- **num_of_accounts**: Number of financial accounts held
- **derogatory_marks**: Number of negative credit marks (e.g., late payments)
- **total_debt**: Total outstanding debt
- **loan_status** (*Target Variable*):
  - `0` = Healthy loan (low risk)
  - `1` = High-risk loan (potential default)

## Model Development
### **1. Data Preprocessing**
- Separated features (`X`) and target variable (`y`)
- Split data into **80% training and 20% testing sets**

### **2. Model Training**
- Used `sklearn.linear_model.LogisticRegression`
- Assigned `random_state=1` for reproducibility
- Trained the model on the training dataset

### **3. Model Evaluation**
#### **Classification Report:**
```
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     15001
           1       0.86      0.94      0.90       507

    accuracy                           0.99     15508
   macro avg       0.93      0.97      0.95     15508
weighted avg       0.99      0.99      0.99     15508
```

#### **Key Insights:**
- ✅ **Excellent performance in predicting `0` (healthy loans)**
- ✅ **Overall accuracy of 99%**
- ⚠️ **Slightly lower precision (0.86) for `1` (high-risk loans), meaning some false positives**

## Future Improvements
- **Feature Engineering**: Identify most important predictors for high-risk loans
- **Automated Model Selection**: Use **TPOT (Tree-based Pipeline Optimization Tool)** to find the best-performing model.
- **Neural Networks**: Implement a **deep learning-based approach** using TensorFlow/Keras for improved accuracy on complex patterns.

## Conclusion
This project successfully implemented **Logistic Regression** to predict loan risk with **high accuracy**. Further improvements can be made to enhance precision for high-risk loans (`1`).

**Future work** can include balancing the dataset, refining features, and trying alternative classification models like **Random Forest or XGBoost**.
