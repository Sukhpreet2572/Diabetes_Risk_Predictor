import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Health Risk Predictor API",
    description="FastAPI service serving a production Random Forest classifier for diabetes risk prediction.",
    version="2.0.0"
)

# Define absolute paths based on the location of main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "models", "imputer.pkl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "models", "test_data.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "evaluation_metrics.csv")

# Standard feature ordering expected by the model
FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
ZERO_IMPUTATION_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Load assets at startup
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Model or Scaler file not found. Please run train.py first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Load optional preprocessing imputer (training column medians)
imputer = None
if os.path.exists(IMPUTER_PATH):
    with open(IMPUTER_PATH, "rb") as f:
        imputer = pickle.load(f)

# Load optional test data for dynamic evaluation metrics
test_data = None
if os.path.exists(TEST_DATA_PATH):
    with open(TEST_DATA_PATH, "rb") as f:
        test_data = pickle.load(f)


def explain_tree_prediction(rf_model, sample_array):
    """
    Computes exact local feature contributions for tree-based models
    using decision path decomposition (Saabas algorithm).
    For binary classification, decomposes the predicted risk probability:
    P(Risk=1) = base_value + sum(contributions)
    """
    sample_f32 = sample_array.astype(np.float32)
    n_features = sample_array.shape[1]
    contributions = np.zeros(n_features, dtype=float)
    base_values = []

    for estimator in rf_model.estimators_:
        tree = estimator.tree_
        node_indicator = tree.decision_path(sample_f32)
        node_index = node_indicator.indices

        # Probability of class 1 at each node
        node_probs = tree.value[:, 0, 1] / tree.value[:, 0].sum(axis=1)
        base_values.append(node_probs[0])

        for i in range(len(node_index) - 1):
            curr_node = node_index[i]
            next_node = node_index[i + 1]
            feat = tree.feature[curr_node]
            diff = node_probs[next_node] - node_probs[curr_node]
            contributions[feat] += diff

    contributions /= len(rf_model.estimators_)
    base_value = float(np.mean(base_values))
    return base_value, contributions


# Request schema matching the Pima Indians dataset variables
class PatientData(BaseModel):
    pregnancies: int = Field(..., ge=0, description="Number of times pregnant")
    glucose: float = Field(..., ge=0, description="Plasma glucose concentration at 2 hours in an oral glucose tolerance test")
    blood_pressure: float = Field(..., ge=0, description="Diastolic blood pressure (mm Hg)")
    skin_thickness: float = Field(..., ge=0, description="Triceps skin fold thickness (mm)")
    insulin: float = Field(..., ge=0, description="2-Hour serum insulin (mu U/ml)")
    bmi: float = Field(..., ge=0, description="Body mass index (weight in kg/(height in m)^2)")
    dpf: float = Field(..., ge=0, description="Diabetes pedigree function (hereditary score)")
    age: int = Field(..., ge=0, description="Age in years")


@app.get("/")
async def root():
    return {
        "service": "Diabetes Health Risk Predictor API",
        "deployed_model": "Random Forest",
        "status": "healthy",
        "version": "2.0.0"
    }


@app.post("/predict")
async def predict_risk(data: PatientData):
    try:
        raw_dict = {
            "Pregnancies": float(data.pregnancies),
            "Glucose": float(data.glucose),
            "BloodPressure": float(data.blood_pressure),
            "SkinThickness": float(data.skin_thickness),
            "Insulin": float(data.insulin),
            "BMI": float(data.bmi),
            "DiabetesPedigreeFunction": float(data.dpf),
            "Age": float(data.age)
        }

        # Apply training median imputation for invalid/missing zero clinical values if imputer is available
        if imputer:
            for col in ZERO_IMPUTATION_COLUMNS:
                if raw_dict[col] == 0.0 and col in imputer:
                    raw_dict[col] = float(imputer[col])

        # Construct DataFrame with exact feature ordering and names
        features_df = pd.DataFrame([raw_dict])[FEATURE_NAMES]

        # Scale features using fitted StandardScaler
        scaled_features = scaler.transform(features_df)

        # Predict binary class (0: Low Risk, 1: High Risk)
        prediction = int(model.predict(scaled_features)[0])

        # Predict class probabilities
        probabilities = model.predict_proba(scaled_features)[0]
        risk_probability = float(probabilities[1])

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

        # Compute tree-based local explanations
        if isinstance(model, RandomForestClassifier):
            base_value, tree_contribs = explain_tree_prediction(model, scaled_features)
            intercept = round(base_value, 4)
            contributions = {
                name: round(float(contrib), 4)
                for name, contrib in zip(FEATURE_NAMES, tree_contribs)
            }
        else:
            # Fallback for linear models if loaded
            coefficients = getattr(model, "coef_", [np.zeros(len(FEATURE_NAMES))])[0]
            intercept = float(getattr(model, "intercept_", [0.0])[0])
            contributions = {
                name: round(float(coef * val), 4)
                for name, coef, val in zip(FEATURE_NAMES, coefficients, scaled_features[0])
            }

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
        if isinstance(model, RandomForestClassifier):
            importances = model.feature_importances_
            feature_imp_map = {name: round(float(imp), 4) for name, imp in zip(FEATURE_NAMES, importances)}
            
            sorted_factors = sorted(
                [
                    {
                        "feature": name,
                        "importance": round(float(imp), 4),
                        "coefficient": round(float(imp), 4),       # backwards compatibility for clients
                        "abs_coefficient": round(float(imp), 4)   # backwards compatibility for clients
                    }
                    for name, imp in zip(FEATURE_NAMES, importances)
                ],
                key=lambda x: x["importance"],
                reverse=True
            )
            
            return {
                "model_type": "Random Forest",
                "intercept": None,
                "feature_importances": feature_imp_map,
                "coefficients": feature_imp_map,  # backwards compatibility
                "top_risk_factors": sorted_factors,
                "status": "success"
            }
        else:
            coefficients = model.coef_[0]
            intercept = float(model.intercept_[0])
            insights = {name: float(coef) for name, coef in zip(FEATURE_NAMES, coefficients)}
            sorted_factors = sorted(
                [
                    {"feature": name, "coefficient": float(coef), "abs_coefficient": float(abs(coef))}
                    for name, coef in zip(FEATURE_NAMES, coefficients)
                ],
                key=lambda x: x["abs_coefficient"],
                reverse=True
            )
            return {
                "model_type": "Logistic Regression",
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

        # Run prediction on test set with currently deployed model
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = map(int, cm.ravel())

        # ROC Curve points
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = float(roc_auc_score(y_test, y_prob))

        return {
            "deployed_model": "Random Forest" if isinstance(model, RandomForestClassifier) else "Logistic Regression",
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
        "model_type": type(model).__name__,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "imputer_loaded": imputer is not None,
        "test_data_loaded": test_data is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
