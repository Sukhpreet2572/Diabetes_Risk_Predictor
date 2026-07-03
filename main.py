import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Initialize FastAPI app
app = FastAPI(title="Diabetes Health Risk Predictor API")

# Define absolute paths based on the location of main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "models", "test_data.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "evaluation_metrics.csv")

# Load assets at startup
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Model or Scaler file not found. Please run train.py first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Optional: Load test data if available for evaluation page
test_data = None
if os.path.exists(TEST_DATA_PATH):
    with open(TEST_DATA_PATH, "rb") as f:
        test_data = pickle.load(f)

# Request schema matching the Pima Indians dataset variables
class PatientData(BaseModel):
    pregnancies: int = Field(..., ge=0, description="Number of times pregnant")
    glucose: float = Field(..., ge=0, description="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    blood_pressure: float = Field(..., ge=0, description="Diastolic blood pressure (mm Hg)")
    skin_thickness: float = Field(..., ge=0, description="Triceps skin fold thickness (mm)")
    insulin: float = Field(..., ge=0, description="2-Hour serum insulin (mu U/ml)")
    bmi: float = Field(..., ge=0, description="Body mass index (weight in kg/(height in m)^2)")
    dpf: float = Field(..., ge=0, description="Diabetes pedigree function (hereditary score)")
    age: int = Field(..., ge=0, description="Age in years")

@app.post("/predict")
async def predict_risk(data: PatientData):
    try:
        # Convert input Pydantic model to list
        raw_features = [
            data.pregnancies,
            data.glucose,
            data.blood_pressure,
            data.skin_thickness,
            data.insulin,
            data.bmi,
            data.dpf,
            data.age
        ]
        
        # Reshape to a 2D array for the scaler (1 sample, 8 features)
        features_array = np.array(raw_features).reshape(1, -1)
        
        # Scale the features using the saved StandardScaler
        scaled_features = scaler.transform(features_array)
        
        # Predict class (0: Low Risk, 1: High Risk)
        prediction = int(model.predict(scaled_features)[0])
        
        # Predict probability
        probabilities = model.predict_proba(scaled_features)[0]
        risk_probability = float(probabilities[1])
        
        # Feature names mapping
        feature_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        
        # Calculate Confidence Score based on the distance from the 0.5 decision boundary
        pred_class_prob = risk_probability if prediction == 1 else (1.0 - risk_probability)
        if pred_class_prob >= 0.90:
            confidence_level = "Very High"
        elif pred_class_prob >= 0.75:
            confidence_level = "High"
        elif pred_class_prob >= 0.60:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
            
        # Calculate Explainability metrics (Log-odds contributions)
        coefficients = model.coef_[0]
        intercept = float(model.intercept_[0])
        
        contributions = {}
        for name, coef, val in zip(feature_names, coefficients, scaled_features[0]):
            contributions[name] = float(coef * val)
            
        return {
            "prediction": prediction,
            "risk_probability": round(risk_probability, 4),
            "confidence_level": confidence_level,
            "intercept": intercept,
            "contributions": contributions,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-insights")
async def get_model_insights():
    try:
        feature_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        coefficients = model.coef_[0]
        intercept = float(model.intercept_[0])
        
        # Combine feature names with coefficients
        insights = {name: float(coef) for name, coef in zip(feature_names, coefficients)}
        
        # Order risk factors by their absolute influence
        sorted_factors = sorted(
            [
                {"feature": name, "coefficient": float(coef), "abs_coefficient": float(abs(coef))} 
                for name, coef in zip(feature_names, coefficients)
            ],
            key=lambda x: x["abs_coefficient"],
            reverse=True
        )
        
        return {
            "intercept": intercept,
            "coefficients": insights,
            "top_risk_factors": sorted_factors,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model insights: {str(e)}")

@app.get("/model-evaluation")
async def get_model_evaluation():
    try:
        # 1. Read comparison table from CSV
        if not os.path.exists(METRICS_PATH):
            raise FileNotFoundError("Metrics CSV file not found. Please run train.py first.")
        
        metrics_df = pd.read_csv(METRICS_PATH)
        comparison_table = metrics_df.to_dict(orient="records")
        
        # 2. Compute dynamic metrics if test data is available
        if test_data is None:
            raise FileNotFoundError("Test data pickle file not found. Please run train.py first.")
            
        X_test_scaled = test_data["X_test_scaled"]
        y_test = test_data["y_test"]
        
        # Run prediction on test set
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = map(int, cm.ravel())
        
        # ROC Curve points
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc_score = float(roc_auc_score(y_test, y_prob))
        
        return {
            "comparison_table": comparison_table,
            "confusion_matrix": {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp
            },
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": round(auc_score, 4)
            },
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model evaluation data: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None, 
        "scaler_loaded": scaler is not None,
        "test_data_loaded": test_data is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
