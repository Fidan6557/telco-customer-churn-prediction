# 📊 Telecom Customer Churn Prediction

This project involves an end-to-end Machine Learning pipeline to predict customer churn for a telecommunications company. It includes deep exploratory data analysis (EDA), advanced feature engineering, and the comparison of multiple boosting algorithms.

## 📌 Project Overview
The goal is to identify customers who are likely to churn (leave the service) by analyzing their usage patterns, contract types, and demographic data. This allows the business to take proactive measures for customer retention.

## 📊 Technologies & Libraries
The project is built with the following stack:
* **Python**: Primary language.
* **Pandas & NumPy**: For data cleaning and matrix operations.
* **Matplotlib & Seaborn**: For statistical data visualization and correlation analysis.
* **Scikit-learn**: For preprocessing (Scaling, Encoding) and model evaluation metrics.
* **Advanced Models**: **XGBoost**, **LightGBM**, and **CatBoost**.
* **Skopt**: Used for **Bayesian Optimization** (`BayesSearchCV`) to find the best hyperparameters.

## 🛠️ Project Workflow
1. **Data Preprocessing**: Handled missing values in `TotalCharges` and converted types.
2. **EDA**: Visualized churn distribution and analyzed categorical feature impacts using countplots.
3. **Feature Engineering**: 
    * Performed **One-Hot Encoding** for categorical features.
    * Used **StandardScaler** to normalize numerical variables.
4. **Modeling**: Compared Baseline Random Forest with Tuned models and Gradient Boosting machines.
5. **Evaluation**: Assessed performance using Accuracy, Precision, Recall, and **ROC-AUC**.

## 📈 Model Performance & Comparison
After hyperparameter tuning and model comparison, the results were as follows:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RandomForest (Tuned)** | 0.8034 | 0.6678 | 0.5160 | 0.5822 | **0.8449** |
| **CatBoost** | 0.8020 | 0.6547 | 0.5374 | 0.5903 | 0.8416 |
| **LightGBM** | 0.7942 | 0.6312 | 0.5401 | 0.5821 | 0.8341 |

### Visual Insights
<img width="536" height="470" alt="download" src="https://github.com/user-attachments/assets/972e23ea-2ba8-4fa1-ba7b-a82cdf331750" />
<img width="952" height="470" alt="download" src="https://github.com/user-attachments/assets/cf79bd18-51b6-4789-b2ff-76209bfcde36" />


* **Feature Importance**: The analysis shows that `tenure`, `Contract_Two year`, and `MonthlyCharges` are the most critical factors influencing a customer's decision to stay or leave.
* **ROC-AUC Curve**: Our best model achieved an AUC of **0.845**, indicating strong predictive power.



## 🚀 Deployment Ready
The final optimized model is saved as `best_model.pkl` using the `joblib` library, making it ready for production use and real-time predictions.

## 📂 Dataset
The dataset used in this project is the **Telco Customer Churn** dataset.
