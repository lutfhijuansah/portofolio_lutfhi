import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Import Dataset
df = pd.read_csv('Loan Default Prediction Dataset.csv')

# 'Default' adalah target, dan 'LoanID' adalah ID
X = df.drop(columns=['LoanID', 'Default'])
y = df['Default']

# Identifikasi kolom kategorikal dan numerik (ini harus sesuai dengan data Anda!)
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
numerical_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

# One-Hot Encoding pada X (pastikan drop_first=True konsisten)
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split data (Stratified untuk imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training model Logistic Regression
logreg = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # Pastikan class_weight sesuai
logreg.fit(X_train_scaled, y_train)

# --- INI ADALAH BAGIAN UTAMA UNTUK DEPLOYMENT ---
# 1. Simpan model terlatih
joblib.dump(logreg, 'logistic_regression_model.pkl')

# 2. Simpan scaler yang sudah fit
joblib.dump(scaler, 'scaler.pkl')

# 3. Simpan nama-nama fitur (order penting!)
feature_names = X_train.columns.tolist() # Ini adalah nama kolom setelah one-hot encoding
joblib.dump(feature_names, 'feature_names.pkl')

# 4. Simpan info kategori unik untuk fitur kategorikal (untuk one-hot encoding di Streamlit)
categorical_features_and_categories = {
    'Education': ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], # Isi dengan kategori unik dari data training Anda
    'EmploymentType': ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], # Isi dengan kategori unik dari data training Anda
    'MaritalStatus': ['Single', 'Married', 'Divorced'], # Isi dengan kategori unik dari data training Anda
    'HasMortgage': ['No', 'Yes'],
    'HasDependents': ['No', 'Yes'],
    'LoanPurpose': ['Business', 'Education', 'Home', 'Other'],
    'HasCoSigner': ['No', 'Yes']
}
joblib.dump(categorical_features_and_categories, 'categorical_info.pkl')

print("Semua file .pkl berhasil disimpan!")