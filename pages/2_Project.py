# pages/2_Project.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # If you want to display plots from code

st.set_page_config(page_title="Project Details", layout="centered")

st.title("Project Details: Loan Default Risk Prediction")
st.markdown("---")

# --- Project Background ---
st.header("🎯 Background / Problem Statement")
st.write("In the banking industry, credit risk assessment is a crucial step to minimize financial losses. This project aims to build a Machine Learning model capable of predicting the probability of a customer defaulting on their loan. This enables the bank to make more informed and proactive decisions.")
st.markdown("---")

# --- Data Exploration (EDA) & Key Visualizations ---
st.header("📊 Data Exploration & Insights")
st.write("This stage involves understanding the data structure, identifying patterns, and discovering initial insights that will guide the modeling process.")

st.subheader("Target Variable Distribution")
# Example: You can display an image you've already created in your notebook
st.image("images/distribusi_target.png", caption="Distribution of Default vs Non-Default")
st.write("From this visualization, it is evident that the data tends to be *imbalanced*, with a significantly higher number of 'non-default' cases compared to 'default' cases.")

# --- Modified Section: st.tabs() ---
st.subheader("Feature Correlation with Default Status")

tab1, tab2 = st.tabs(["Numerical Features", "Categorical Features"])

with tab1:
    st.write("Visualization of the relationship between numerical features and default status.")
    st.image("images/numerikal_default.png", caption="Boxplot: Numerical Features vs Default Status") # Replace with your income boxplot image
    st.write("Customers who default are generally younger, have lower income and shorter employment duration, and slightly higher loan amounts. Although credit scores and DTI ratios do not show striking differences, credit scores in the default group tend to be slightly lower.")

with tab2:
    st.write("Visualization of the relationship between categorical features and default status.")
    st.image("images/kategorikal_default.png", caption="Countplot: Categorical Features vs Default Status") # Replace with your education countplot image
    st.write("The risk of default tends to be lower for borrowers with higher education, permanent employment, married status, a mortgage, dependents, and a co-signer. Conversely, a higher risk is found among unemployed borrowers, those working part-time, or those taking loans for business purposes.")

st.markdown("---")

# --- Modeling Methodology ---
st.header("🛠️ Modeling Methodology")
st.write("The modeling process involves several key stages:")
st.markdown("""
- **Data Preprocessing:** Handling *missing values*, *outliers*, *one-hot encoding* for categorical variables (ensuring consistency with training), and *Standard Scaling* for numerical variables.
- **Feature Selection:** Identifying the most relevant features based on correlation analysis and domain knowledge.
- **Model Selection:** We evaluated several classification models such as Logistic Regression, Random Forest, and XGBoost. **Logistic Regression** was chosen as the primary model due to its good interpretability and stable performance on this dataset.
- **Handling Imbalanced Data:** Due to the fewer number of 'default' cases, we used `class_weight='balanced'` in Logistic Regression to ensure the model is not biased towards the majority class.
""")
st.markdown("---")

# --- Model Results & Evaluation ---
st.header("📈 Model Evaluation Results")
st.write("The Logistic Regression model was evaluated using the *test set* to ensure good generalization.")

st.subheader("Model Performance (Logistic Regression)")
st.image("images/logistic_classification_report.png", caption="Model Performance Metrics on Test Set") #
st.write("The table above shows the performance of the Logistic Regression model on the *train set* and *test set*. We focused on Recall and F1-score for the 'default' class given the *imbalanced class* problem.")

st.subheader("Precision-Recall Curve")
st.image("images/precision_recall.png", caption="Precision-Recall Curve with Optimal Threshold") #
st.write(f"Based on the Precision-Recall Curve, the optimal `threshold` was chosen at `0.6231` to balance `Precision` (`0.2879`) and `Recall` (`0.5087`), resulting in the best `F1-score` (`0.3677`).")

st.subheader("Confusion Matrix (Test Set)")
st.image("images/confusssion_matrix.png", caption="Confusion Matrix for Logistic Regression") #
st.write("The Confusion Matrix shows that out of 5,931 'Actual Default' cases, the model correctly predicted 3,016 cases (True Positives), but there were still 2,915 cases incorrectly predicted as 'Non-Default' (False Negatives).")

st.subheader("Feature Interpretation (Coefficients)")
st.image("images/feature_importance.png", caption="Top 10 Most Important Features in Logistic Regression") #
st.write("The Logistic Regression model is interpretable. Some important features and their influences include:")
st.markdown("""
- **Age, MonthsEmployed, Income, CreditScore, HasCoSigner_Yes, HasDependents_Yes:** These have negative coefficients, indicating that higher values in these features (or `Yes` for HasCoSigner/HasDependents) tend to decrease the risk of default.
- **InterestRate, LoanAmount, EmploymentType_Unemployed, EmploymentType_Part-time:** These have positive coefficients, indicating that higher values in these numerical features or these employment categories tend to increase the risk of default.
""")
st.markdown("---")

st.header("💡 Conclusion & Future Development")
st.write("This project successfully built and implemented a predictive model for loan default risk, providing a valuable tool for banks in credit decision-making. The Logistic Regression model demonstrated a good balance between interpretability and performance.")
st.write("For future development, this project can be enhanced by:")
st.markdown("""
- **Additional Data:** Incorporating more features (e.g., previous payment history, external data such as macroeconomic indicators).
- **Advanced Models:** Exploring *ensemble* techniques or *deep learning* if the data becomes more complex.
- **Model Monitoring:** Developing a system to monitor model performance periodically in a production environment.
- **A/B Testing:** Testing the direct impact of using this model on the bank's business decisions.
""")