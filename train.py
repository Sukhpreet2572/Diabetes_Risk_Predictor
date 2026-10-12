import os
import pickle
import numpy as np
import pandas as pd
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
    # 1. Ensure required directories exist
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
    
    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    target_column = "Outcome"
    
    X = df[feature_names].copy()
    y = df[target_column].copy()
    
    # 3. Train-Test Split (80% train, 20% test, stratified)
    # Split BEFORE preprocessing to strictly prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # 4. Data Cleaning / Imputation fitted strictly on training data
    # Physiologically impossible zero values in clinical indicators
    zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    print("\n--- Leakage-Free Preprocessing ---")
    print("Computing column medians on training split:")
    train_medians = {}
    for col in zero_columns:
        train_valid_vals = X_train[col].replace(0, np.nan)
        median_val = float(train_valid_vals.median())
        train_medians[col] = median_val
        zeros_in_train = int((X_train[col] == 0).sum())
        zeros_in_test = int((X_test[col] == 0).sum())
        print(f"  - {col}: median = {median_val:.2f} (Train zeros: {zeros_in_train}, Test zeros: {zeros_in_test})")
        
    # Impute missing zero values in train and test using training medians
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    for col in zero_columns:
        X_train_imputed[col] = X_train_imputed[col].replace(0, np.nan).fillna(train_medians[col])
        X_test_imputed[col] = X_test_imputed[col].replace(0, np.nan).fillna(train_medians[col])
        
    # Save fitted medians / imputer artifact for production inference
    imputer_path = os.path.join("models", "imputer.pkl")
    with open(imputer_path, "wb") as f:
        pickle.dump(train_medians, f)
    print(f"Fitted training medians saved to {imputer_path}")
    
    # 5. Feature Scaling (Fitted exclusively on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    scaler_path = os.path.join("models", "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Fitted StandardScaler saved to {scaler_path}")
    
    # Save test data for dynamic evaluation endpoint
    test_data_path = os.path.join("models", "test_data.pkl")
    with open(test_data_path, "wb") as f:
        pickle.dump({
            "X_test_scaled": X_test_scaled,
            "y_test": y_test.values,
            "feature_names": feature_names
        }, f)
    print(f"Test data saved to {test_data_path}")
    
    # 6. Model Training & Comparison
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    if XGB_AVAILABLE:
        print("XGBoost is available. Training XGBClassifier...")
        models["XGBoost"] = xgb.XGBClassifier(
            random_state=42, 
            eval_metric="logloss"
        )
    else:
        print("XGBoost is not available. Training GradientBoostingClassifier (scikit-learn) as fallback...")
        models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)
        
    metrics_list = []
    confusion_matrices = {}
    
    print("\n--- Training Models & Calculating Metrics ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Compute Binary Classification Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm
        
        metrics_list.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })
        
    # Deploy Random Forest as the primary model
    deployed_model_name = "Random Forest"
    deployed_model = models[deployed_model_name]
    model_path = os.path.join("models", "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(deployed_model, f)
    print(f"\n[DEPLOYMENT] Deployed {deployed_model_name} saved to {model_path}")
    
    # 7. Create metrics comparison table and save to CSV
    metrics_df = pd.DataFrame(metrics_list)
    print("\n--- Model Comparison Table (Holdout Test Split: 20%) ---")
    print(metrics_df.to_string(index=False))
    
    print("\n--- Confusion Matrices ---")
    for name, cm in confusion_matrices.items():
        tn, fp, fn, tp = cm.ravel()
        print(f"{name}: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
    csv_metrics_path = os.path.join("models", "evaluation_metrics.csv")
    metrics_df.to_csv(csv_metrics_path, index=False)
    print(f"\nEvaluation metrics saved to {csv_metrics_path}")
    print(f"\nSelected Deployed Model: {deployed_model_name}")


if __name__ == "__main__":
    main()
