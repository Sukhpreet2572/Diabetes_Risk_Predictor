# Diabetes Health Risk Predictor

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-11754C?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)

An interactive, production-grade machine learning application designed to assess a patient's risk of diabetes based on standard clinical indicators from the Pima Indians Diabetes Dataset. The project follows a decoupled microservice architecture with a **FastAPI** backend service and a multi-page **Streamlit** diagnostic dashboard.

Following rigorous multi-model comparison, the project deploys a **Random Forest Classifier (100 estimators)** as its primary production model, offering superior accuracy, recall, and discrimination capability over linear baselines.

---

## Architecture Overview

```
                          [ Raw CSV Dataset ]
                                   │
                                   ▼
             [ Leakage-Free Preprocessing Pipeline ]
             ├─ Split: Stratified 80/20 train-test partition
             ├─ Missing Imputation: Column medians fitted strictly on train
             └─ Normalization: StandardScaler fitted strictly on train
                                   │
                                   ▼
      ┌────────────────────────────┼────────────────────────────┐
      │                            │                            │
      ▼                            ▼                            ▼
[ Random Forest ]            [ XGBoost ]             [ Logistic Regression ]
 (Deployed Model)         (Comparison Model)            (Baseline Model)
  Acc: 77.27%                Acc: 76.62%                  Acc: 70.78%
  AUC: 0.8181                AUC: 0.8102                  AUC: 0.8130
      │
      ├─► Exports models/model.pkl, scaler.pkl, imputer.pkl, test_data.pkl
      │
      ▼
[ FastAPI Backend ] (Port 8001)
      │
      ├─► GET  /                 (Service info & deployed model)
      ├─► GET  /health           (Liveness & artifact validation)
      ├─► POST /predict          (Local decision-path attribution + confidence)
      ├─► GET  /model-insights   (Global Gini ensemble feature importance)
      ├─► GET  /model-evaluation (Dynamic confusion matrix & ROC curve metrics)
      │
      ▼
[ Streamlit Dashboard ] (Port 8501) ◄─► Clinician / User Interface
```

### Key Architectural Decisions: Pipeline vs. Decoupled Artifacts
This repository uses decoupled artifact persistence (`model.pkl`, `scaler.pkl`, `imputer.pkl`) rather than a single unified `sklearn.pipeline.Pipeline` object. 
- **Decoupled Trade-off:** Allows independent inspection, dynamic test evaluation (`models/test_data.pkl`), and modular reuse of the scaler and imputer without reloading tree ensembles. 
- **Pipeline Alternative:** Bundling preprocessing and estimator into a unified Pipeline simplifies serialization to a single `.pkl` file. However, keeping decoupled artifacts maintains 100% backward compatibility with existing backend initialization scripts and health-check monitoring probes.

---

## Evaluated Models & Production Benchmark

All models were evaluated on the exact same holdout testing set (stratified 20% split, 154 test samples) using identical training preprocessing:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🌲 **Random Forest (Deployed)** | **0.7727** | **0.7021** | **0.6111** | **0.6535** | **0.8181** |
| ⚡ **XGBoost** | 0.7662 | 0.6875 | 0.6111 | 0.6471 | 0.8081 |
| 📈 **Logistic Regression (Baseline)** | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 |

### Holdout Confusion Matrix (Random Forest):
- **True Negatives (TN):** 86 (Correctly predicted low risk)
- **False Positives (FP):** 14 (Healthy flagged for confirmatory screening)
- **False Negatives (FN):** 21 (Diabetic cases missed)
- **True Positives (TP):** 33 (Diabetic cases correctly identified)

---

## Setup & Execution Guide

### 1. Environment Installation
Install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```
*(Or install manually: `pip install fastapi uvicorn streamlit pandas numpy scikit-learn matplotlib requests xgboost`)*

### 2. Model Training & Leakage-Free Preprocessing
Run the training script to fetch the dataset, fit training medians, scale features, evaluate all three candidate models, and deploy the Random Forest model:
```bash
python train.py
```
This saves:
- `models/model.pkl` — Fitted production `RandomForestClassifier`
- `models/scaler.pkl` — Fitted `StandardScaler`
- `models/imputer.pkl` — Fitted training medians for invalid zero imputation
- `models/test_data.pkl` — Holdout test partition for dynamic evaluations
- `models/evaluation_metrics.csv` — Comparative metrics table

### 3. Start FastAPI Microservice
Launch the backend server on port 8001:
```bash
python main.py
```
*(Or directly with uvicorn: `uvicorn main:app --host 127.0.0.1 --port 8001 --reload`)*

### 4. Launch Streamlit UI
In a separate terminal window, start the frontend dashboard:
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

### 5. Running the Automated Test Suite
Execute the unit and integration tests:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## API Documentation & Endpoints

### 1. `POST /predict`
Evaluates a patient's risk profile and returns classification, confidence level, and local decision-path feature contributions.

**Request Body (`application/json`):**
```json
{
  "pregnancies": 2,
  "glucose": 138.0,
  "blood_pressure": 74.0,
  "skin_thickness": 26.0,
  "insulin": 110.0,
  "bmi": 28.5,
  "dpf": 0.52,
  "age": 36
}
```

**Response (`200 OK`):**
```json
{
  "prediction": 0,
  "risk_probability": 0.28,
  "confidence_level": "Medium",
  "intercept": 0.3466,
  "contributions": {
    "Pregnancies": 0.0125,
    "Glucose": 0.0450,
    "BloodPressure": -0.0102,
    "SkinThickness": -0.0054,
    "Insulin": -0.0180,
    "BMI": -0.0725,
    "DiabetesPedigreeFunction": -0.0060,
    "Age": -0.0120
  },
  "status": "success"
}
```

### 2. `GET /model-insights`
Returns global Gini feature importance (Mean Decrease in Impurity) across the 100 decision trees.

### 3. `GET /model-evaluation`
Returns the multi-model comparison table, confusion matrix counts, and ROC curve coordinates for the deployed model.

### 4. `GET /health`
Returns system health, confirming that `RandomForestClassifier`, the scaler, and preprocessing objects are loaded in memory.

---

## Explainability: Global Importance vs. Local Attribution

1. **Global Feature Importance (Gini / MDI):**
   - Summarizes macro ensemble behavior across all splits in the 100 trees.
   - Identified **Glucose** (~27.3%), **BMI** (~16.4%), and **Diabetes Pedigree Function** (~12.5%) as the most decisive overall risk indicators.
2. **Local Prediction Attribution (Decision Path Decomposition):**
   - Implements the Saabas algorithm tracking the exact probability delta ($\Delta P = P_{child} - P_{parent}$) at each split along the tree paths for that specific patient.
   - Satisfies the exact additive property:
     $$\text{Risk Probability} = \text{Baseline Prior} + \sum_{i=1}^{8} \text{Contribution}_i$$
   - Clearly separates general risk factors from patient-specific drivers.

---

## Clinical Disclaimer & Medical Prototype Limitations

> [!WARNING]
> **Prototype Demonstration Notice:**
> - This software is an engineering demonstration prototype and **is NOT a clinically validated medical device or diagnostic instrument**.
> - The model was trained on the Pima Indians Diabetes Dataset, which reflects a specific demographic subset (adult females of Pima Indian heritage).
> - Predictions must never be used as a substitute for professional medical advice, clinical judgment, laboratory testing (such as fasting blood glucose or HbA1c), or diagnosis by a licensed healthcare provider.
