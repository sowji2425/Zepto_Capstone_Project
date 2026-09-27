import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

DATA_PATH = "analytics/customer_data.csv"

def generate_mock_data():
    """Generates a reproducible dataset simulating customer metrics."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        "customer_id": range(10001, 10001 + n_samples),
        "tenure_months": np.random.randint(1, 72, size=n_samples),
        "monthly_spend_inr": np.random.uniform(500, 8000, size=n_samples),
        "support_tickets": np.random.randint(0, 10, size=n_samples),
        "payment_method": np.random.choice(["Credit Card", "UPI", "Net Banking"], size=n_samples)
    }
    
    df = pd.DataFrame(data)
    # Define a deterministic target rule with noise for churn prediction
    churn_prob = (df["support_tickets"] * 0.15) - (df["tenure_months"] * 0.005) + (df["monthly_spend_inr"] * 0.00005)
    df["churned"] = (churn_prob > np.percentile(churn_prob, 70)).astype(int)
    
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset generated and cached at {DATA_PATH}")

def run_pipeline():
    if not os.path.exists(DATA_PATH):
        generate_mock_data()
        
    df = pd.read_csv(DATA_PATH)
    
    # --- PHASE 1: Data Profiling & Prep ---
    print("\n--- Summary Statistics ---")
    print(df.describe().T)
    
    print("\n--- Target Class Distribution ---")
    print(df["churned"].value_counts(normalize=True))
    
    # Feature Engineering & Encoding
    X = df[["tenure_months", "monthly_spend_inr", "support_tickets", "payment_method"]].copy()
    X = pd.get_dummies(X, columns=["payment_method"], drop_first=True)
    y = df["churned"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical values
    scaler = StandardScaler()
    num_cols = ["tenure_months", "monthly_spend_inr", "support_tickets"]
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # --- PHASE 2: Modeling ---
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    clf.fit(X_train, y_train)
    
    # Evaluative Scoring
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation Summary ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Feature Importances
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Top Predictive Factors ---")
    print(importances)

if __name__ == "__main__":
    run_pipeline()
