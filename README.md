# Diabetes Health Risk Predictor

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An interactive, production-style machine learning application designed to assess a patient's risk of diabetes based on standard clinical metrics. The project separates concerns into a backend FastAPI microservice and a multi-page Streamlit UI dashboard, illustrating how to deploy and explain machine learning models in a decoupled architecture.

---

## Architecture Overview

```
                          [ Raw CSV Dataset ]
                                  │
                                  ▼
                         [ Data Preprocessing ] (Imputation & Scaling)
                                  │
                                  ▼
     ┌────────────────────────────┼────────────────────────────┐
     │                            │                            │
     ▼                            ▼                            ▼
[ Logistic Regression ]     [ Random Forest ]             [ XGBoost ]
  (Primary Model)           (Comparison Model)        (Comparison Model)
     │
     ├─► Exports models/model.pkl & models/scaler.pkl
     │
     ▼
[ FastAPI Backend ] (Port 8001)
     │
     ├─► POST /predict          (Local explainability + confidence evaluation)
     ├─► GET /model-insights    (Global coefficient rankings)
     ├─► GET /model-evaluation  (Dynamic confusion matrix & ROC curve metrics)
     │
     ▼
[ Streamlit Dashboard ] (Port 8501) ◄─► User Interface
```

*   **Model Pipeline:** Reads clinical inputs from the Pima Indians Diabetes Dataset, imputes invalid zero-values (using median value imputation for physiologically impossible fields like Glucose or BMI), and applies `StandardScaler` to normalize feature distributions.
*   **Model Options:** Trains Logistic Regression, Random Forest, and XGBoost/Gradient Boosting models, saving comparative metrics.
*   **Production Deployment:** Deploys the Logistic Regression model. The backend calculates the exact mathematical contribution ($\beta_i \times x^{scaled}_i$) of each vital statistic to explain the prediction outcome.

---

## Deployed Model Performance

Calculated on the holdout testing set (20% of the dataset):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Deployed)** | **0.7078** | **0.6000** | **0.5000** | **0.5455** | **0.8130** |
| **Random Forest** | 0.7792 | 0.7174 | 0.6111 | 0.6600 | 0.8179 |
| **XGBoost** | 0.7597 | 0.6735 | 0.6111 | 0.6408 | 0.8081 |

---

## Setup & Running Guide

### 1. Installation
Install the required packages using pip:
```bash
pip install fastapi uvicorn streamlit pandas numpy scikit-learn matplotlib requests xgboost
```

### 2. Model Training
Run the training pipeline script to process the dataset, generate comparative metrics, and output the model/scaler artifacts:
```bash
python train.py
```

### 3. Run FastAPI Backend
Start the uvicorn web server to host the API endpoints on port 8001:
```bash
python main.py
```

### 4. Run Streamlit UI
In a separate terminal window, launch the interactive user interface:
```bash
streamlit run app.py
```

---

## Core Features

*   **Multi-Page Interface:** Includes structured tabs for `Risk Predictor` (input fields & diagnostic report), `Model Insights` (global variables importance), and `Model Evaluation` (confusion matrix heatmap and ROC curve charts).
*   **Confidence Levels:** Classes are rated based on decision boundary distance (Very High, High, Medium, Low confidence) instead of simple raw classifications.
*   **Input Warnings:** Features validation boundaries to warn users if clinical statistics exceed standard physiological limits (e.g. BMI > 70, Glucose > 400).
*   **Personalized Recommendation Engine:** Inspects the individual vital parameters to formulate custom health guidance tips (e.g., cardiovascular health pointers for high blood pressure or insulin resistance advisories).
