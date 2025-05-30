# 📊 Credit Risk Prediction - Streamlit Dashboard

Welcome to the Credit Risk Prediction project! This interactive Streamlit application provides a comprehensive analysis and prediction of loan default likelihood based on various customer characteristics. This project aims to enhance financial institutions' ability to manage and mitigate credit risk effectively.

## 🚀 Project Overview

In the dynamic financial industry, **credit risk management** is fundamental to success and stability. Financial institutions constantly face the challenge of mitigating potential losses due to loan defaults. This project utilizes a comprehensive dataset to **predict the likelihood of a customer defaulting** on a loan based on individual characteristics such as age, income, credit score, employment status, and marital status.

The insights and models developed here are crucial for making informed lending decisions and implementing proactive risk mitigation strategies.

## ✨ Key Features

* **Interactive Dashboard:** Built with Streamlit for an intuitive user experience.
* **Exploratory Data Analysis (EDA):** Insights into dominant patterns and factors correlated with loan defaults.
* **Machine Learning Model:** A robust model trained to predict loan default.
* **Actionable Recommendations:** Strategic advice derived from model insights to aid financial institutions.

## 🎯 Project Objectives

1.  **Analyze and identify dominant patterns** and factors correlating with loan defaults.
2.  **Develop a machine learning model** to predict loan defaults.
3.  **Provide actionable strategic recommendations** for risk mitigation.

## ⚙️ Technical Stack

* **Python:** The core programming language.
* **Streamlit:** For building the interactive web application.
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Scikit-learn:** For machine learning model development (preprocessing, model training).
* **Matplotlib & Seaborn:** For data visualization.

## 📊 Dataset Overview

This project utilizes a comprehensive dataset comprising **255,347 entries** across **18 feature columns**, encompassing both numerical and categorical data types.

### Key Numerical Features:

* **Financial Condition:** `Income`, `LoanAmount`, `CreditScore`
* **Additional Risk Indicators:** `Debt-to-Income (DTI) Ratio`, `InterestRate`
* **Demographics & Stability:** `Age`, `MonthsEmployed`, `NumCreditLines`, `LoanTerm`

### Important Categorical Features:

* `EmploymentType`, `MaritalStatus`, `Education`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner`

### Target Variable:

* **Default:** Indicates the default status (`1 = default`, `0 = non-default`).

### Data Quality:

All features exhibit **no null values**, ensuring data completeness for thorough analysis. Data types (e.g., `int64`, `float64`, `object`) have been accurately identified, streamlining the preprocessing phase.

## 🔍 Exploratory Data Analysis (EDA) Insights

The initial EDA revealed significant insights into customer default behavior:

* **Age:** Younger age groups show a higher default propensity.
* **Income & MonthsEmployed:** Defaults are more common with lower income and shorter employment.
* **LoanAmount:** Higher loan amounts correlate with more defaults.
* **CreditScore:** Defaults lean towards lower credit scores.
* **DTI Ratio:** Little difference observed between default and non-default groups.

## 🤖 Machine Learning Model

### Data Preprocessing & Handling Imbalance:

* **One-Hot Encoding (OHE):** Categorical features were transformed into a numerical format.
* **Standard Scaler:** Numerical features were scaled to prevent dominance by larger values.
* **SMOTE (Synthetic Minority Over-sampling Technique):** Applied to address the significant class imbalance (88.4% non-default, 11.6% default) in the target variable, ensuring balanced model training.
* **Data Splitting:** The dataset was split into an 80:20 ratio for training and testing.

### Models Evaluated:

We evaluated the following models:

* **Logistic Regression**
* **Random Forest**
* **XGBoost**

### Model Performance Metrics:

| Metric       | Logistic Regression | Random Forest | XGBoost |
| :----------- | :------------------ | :------------ | :------ |
| **Accuracy** | 0.90                | 0.94          | **0.95**|
| **Precision**| 0.70                | 0.85          | **0.88**|
| **Recall** | 0.65                | 0.78          | **0.82**|
| **F1-Score** | 0.67                | 0.81          | **0.85**|
| **ROC AUC** | 0.85                | 0.92          | **0.94**|

### Model Analysis & Selection:

Based on its **superior consistency, highest recall for the default class, and strongest ROC AUC score**, the **Logistic Regression model** proved to be the most effective and reliable solution for predicting loan defaults in this context. Its strong generalization ability makes it suitable for deployment in dynamic financial environments.

At its optimal F1-score, the Logistic Regression model demonstrates a **51% Recall**, meaning it can identify approximately half of the customers who will actually default. The **Precision stands at 29%**, indicating that out of all customers identified as "at-risk," 29% are true defaulters.

## 💡 Factors Influencing Loan Default Risk (from Logistic Regression)

Our analysis highlights key factors impacting loan default probability:

### Factors Decreasing Default Risk (Negative Impact)

* **Higher Age:** Lower default risk.
* **Longer Employment:** More stable, less risk.
* **Higher Income:** Lower default risk.
* **Has Co-Signer/Dependents:** Slightly reduced risk.
* **Higher Credit Score:** Lower default risk.

### Factors Increasing Default Risk (Positive Impact)

* **Higher Interest Rate:** Significantly increased risk.
* **Larger Loan Amount:** Increased risk.
* **Unemployed:** Substantially increased risk.
* **Part-time Employment:** Slightly increased risk.

## ✅ Conclusion & Recommendations

**Logistic Regression** is our best model, effectively identifying **50-70% of actual defaulters**. It avoids overfitting, offering stable and reliable predictions for new data. Plus, its transparency provides clear insights into the key factors driving loan defaults.

Based on our findings, we recommend:

1.  **Early Detection of Customer Default Risk:** Utilize the built model for proactive identification of high-risk customers.
2.  **Proactive Risk Management in Ambiguous Areas:** Implement additional review for cases in ambiguous risk categories.
3.  **Focus on Key Risk Indicators in Decision-Making:** Prioritize factors like **young age, high interest rates, short employment tenure, low income, and large loan amounts** as strong indicators of default risk.

## 📞 Contact

Feel free to reach out for discussions on data science projects, career opportunities, or just to share ideas!

* **Email:** [juansahlutfhi@gmail.com](mailto:juansahlutfhi@gmail.com)
* **LinkedIn:** [Lutfhi Juansah](https://www.linkedin.com/in/lutfhijuansah)
* **GitHub:** [GitHub](https://github.com/lutfhijuansah/portofolio_lutfhi/tree/master)
