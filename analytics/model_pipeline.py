import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

LOCAL_PATH = "analytics/titanic.csv"

def run_model_pipeline():
    # 1. Load the dataset cached by data_profiler.py
    if not os.path.exists(LOCAL_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {LOCAL_PATH}. Please run data_profiler.py first!"
        )
    
    df = pd.read_csv(LOCAL_PATH)
    print("--- Local Dataset Loaded Successfully for Modeling ---")
    print(f"Initial Dimensions: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # 2. Advanced Feature Engineering & Missing Value Imputation
    # Extract passenger titles to dynamically fill missing ages accurately
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_medians = df.groupby('Title')['Age'].transform('median')
    df['Age'] = df['Age'].fillna(title_medians).fillna(df['Age'].median())
    
    # Fill sparse Embarked records using the calculation mode (most common port)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Combine related parameters into a structural interaction feature tracking family units
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Drop structural text identifiers and columns with extreme sparsity (>75% missing like Cabin)
    X = df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Title'])
    y = df['Survived']
    
    # 3. Categorical Conversion & One-Hot Encoding
    X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)
    
    # 4. Stratified Train/Test Split (80/20 Balance Rule)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Continuous Numerical Feature Scaling
    scaler = StandardScaler()
    scale_cols = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])
    
    # 6. Model Training (Random Forest Classifier)
    print("Training Random Forest Classifier model...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    
    # 7. Model Evaluative Scoring
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print("\n=================== CLASSIFICATION REPORT ===================")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Performance Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    print("\n=================== CONFUSION MATRIX ===================")
    print(confusion_matrix(y_test, y_pred))
    
    # 8. Feature Importance Analysis
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=================== FEATURE IMPORTANCE RANKING ===================")
    print(importances)

if __name__ == "__main__":
    run_model_pipeline()
