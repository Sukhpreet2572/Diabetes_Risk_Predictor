import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)

# Attempt to import XGBoost, otherwise fall back to scikit-learn's GradientBoosting
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

def main():
    # 1. Create directories for data and models if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    csv_path = os.path.join("data", "diabetes.csv")
    dataset_url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
    
    # 2. Download/Load the dataset
    if not os.path.exists(csv_path):
        print(f"Downloading dataset from {dataset_url}...")
        df = pd.read_csv(dataset_url)
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")
    else:
        print(f"Loading dataset locally from {csv_path}...")
        df = pd.read_csv(csv_path)
        
    print(f"Dataset shape: {df.shape}")
    
    # 3. Data Cleaning (Impute zero values that represent missing data)
    zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    print("\n--- Preprocessing ---")
    print("Checking for invalid zero values:")
    for col in zero_columns:
        zeros_count = (df[col] == 0).sum()
        print(f"  - {col}: {zeros_count} invalid zero values found.")
        
    # Replace zeros with NaN and then fill with median
    for col in zero_columns:
        df[col] = df[col].replace(0, np.nan)
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        
    print("Invalid zero values successfully replaced with column medians.")
    
    # 4. Split Features (X) and Target (y)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    # 5. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # 6. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the fitted scaler
    scaler_path = os.path.join("models", "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Fitted StandardScaler saved to {scaler_path}")
    
    # Save test data for dynamic metric evaluation in FastAPI
    test_data_path = os.path.join("models", "test_data.pkl")
    with open(test_data_path, "wb") as f:
        pickle.dump({
            "X_test_scaled": X_test_scaled,
            "y_test": y_test.values,
            "feature_names": list(X.columns)
        }, f)
    print(f"Test data saved to {test_data_path}")
    
    # 7. Model Training & Comparison
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    if XGB_AVAILABLE:
        print("XGBoost is available. Training XGBClassifier...")
        models["XGBoost"] = xgb.XGBClassifier(
            random_state=42, 
            eval_metric="logloss", 
            use_label_encoder=False
        )
    else:
        print("XGBoost is not available. Training GradientBoostingClassifier (scikit-learn) as fallback...")
        models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)
        
    metrics_list = []
    
    print("\n--- Training Models & Calculating Metrics ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Compute Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        metrics_list.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })
        
        # Keep Logistic Regression saved as primary model for explainability
        if name == "Logistic Regression":
            model_path = os.path.join("models", "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Logistic Regression saved to {model_path}")
            
    # 8. Create metrics comparison table and save to CSV
    metrics_df = pd.DataFrame(metrics_list)
    print("\n--- Model Comparison Table ---")
    print(metrics_df.to_string(index=False))
    
    csv_metrics_path = os.path.join("models", "evaluation_metrics.csv")
    metrics_df.to_csv(csv_metrics_path, index=False)
    print(f"\nEvaluation metrics saved to {csv_metrics_path}")
    
if __name__ == "__main__":
    main()
