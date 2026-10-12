import os
import unittest
import pickle
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import main
from main import app, FEATURE_NAMES, ZERO_IMPUTATION_COLUMNS


class TestDiabetesRiskPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.repo_dir = os.path.dirname(cls.base_dir)

    def test_model_artifact_is_random_forest(self):
        """Verify deployed artifact is RandomForestClassifier and NOT LogisticRegression."""
        model_path = os.path.join(self.repo_dir, "models", "model.pkl")
        self.assertTrue(os.path.exists(model_path), "models/model.pkl must exist")
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        self.assertIsInstance(
            loaded_model, 
            RandomForestClassifier,
            f"Expected RandomForestClassifier, got {type(loaded_model)}"
        )
        self.assertNotIsInstance(
            loaded_model, 
            LogisticRegression,
            "Deployed model must not be LogisticRegression"
        )
        self.assertTrue(hasattr(loaded_model, "feature_importances_"))
        self.assertEqual(len(loaded_model.feature_importances_), len(FEATURE_NAMES))

    def test_preprocessing_artifacts_consistency(self):
        """Verify scaler, imputer, and test_data artifacts match feature dimensions."""
        scaler_path = os.path.join(self.repo_dir, "models", "scaler.pkl")
        imputer_path = os.path.join(self.repo_dir, "models", "imputer.pkl")
        test_data_path = os.path.join(self.repo_dir, "models", "test_data.pkl")
        metrics_path = os.path.join(self.repo_dir, "models", "evaluation_metrics.csv")

        self.assertTrue(os.path.exists(scaler_path), "scaler.pkl must exist")
        self.assertTrue(os.path.exists(imputer_path), "imputer.pkl must exist")
        self.assertTrue(os.path.exists(test_data_path), "test_data.pkl must exist")
        self.assertTrue(os.path.exists(metrics_path), "evaluation_metrics.csv must exist")

        with open(test_data_path, "rb") as f:
            td = pickle.load(f)
        self.assertEqual(td["feature_names"], FEATURE_NAMES)
        self.assertEqual(td["X_test_scaled"].shape[1], len(FEATURE_NAMES))

    def test_health_and_root_endpoints(self):
        """Verify /health and / endpoints return healthy status and Random Forest model type."""
        root_resp = self.client.get("/")
        self.assertEqual(root_resp.status_code, 200)
        self.assertEqual(root_resp.json()["deployed_model"], "Random Forest")

        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        data = health_resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["model_type"], "RandomForestClassifier")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["scaler_loaded"])
        self.assertTrue(data["imputer_loaded"])

    def test_prediction_class_0_low_risk(self):
        """Verify successful prediction of low-risk case (Class 0)."""
        low_risk_payload = {
            "pregnancies": 0,
            "glucose": 82.0,
            "blood_pressure": 64.0,
            "skin_thickness": 18.0,
            "insulin": 45.0,
            "bmi": 21.5,
            "dpf": 0.18,
            "age": 22
        }
        resp = self.client.post("/predict", json=low_risk_payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()

        self.assertEqual(body["status"], "success")
        self.assertEqual(body["prediction"], 0)
        self.assertLess(body["risk_probability"], 0.50)
        self.assertIn(body["confidence_level"], ["Low", "Medium", "High", "Very High"])
        self.assertIn("contributions", body)
        self.assertEqual(set(body["contributions"].keys()), set(FEATURE_NAMES))

    def test_prediction_class_1_high_risk(self):
        """Verify successful prediction of high-risk case (Class 1)."""
        high_risk_payload = {
            "pregnancies": 9,
            "glucose": 190.0,
            "blood_pressure": 95.0,
            "skin_thickness": 42.0,
            "insulin": 280.0,
            "bmi": 43.0,
            "dpf": 1.45,
            "age": 52
        }
        resp = self.client.post("/predict", json=high_risk_payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()

        self.assertEqual(body["status"], "success")
        self.assertEqual(body["prediction"], 1)
        self.assertGreater(body["risk_probability"], 0.50)
        self.assertIn("contributions", body)

    def test_probability_range_and_explanation_math(self):
        """Verify probability is bounded [0, 1] and decision-path attribution is consistent."""
        sample_payload = {
            "pregnancies": 2,
            "glucose": 125.0,
            "blood_pressure": 72.0,
            "skin_thickness": 22.0,
            "insulin": 90.0,
            "bmi": 27.5,
            "dpf": 0.45,
            "age": 35
        }
        resp = self.client.post("/predict", json=sample_payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()

        prob = body["risk_probability"]
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

        # Check that base_rate + sum(contributions) matches predicted probability closely
        intercept = body["intercept"]
        contrib_sum = sum(body["contributions"].values())
        reconstructed_prob = intercept + contrib_sum
        self.assertAlmostEqual(reconstructed_prob, prob, delta=0.03)

    def test_invalid_and_missing_inputs(self):
        """Verify API handles missing fields and out-of-boundary constraints gracefully."""
        # 1. Missing required field 'glucose'
        incomplete_payload = {
            "pregnancies": 1,
            "blood_pressure": 70.0,
            "skin_thickness": 20.0,
            "insulin": 80.0,
            "bmi": 25.0,
            "dpf": 0.5,
            "age": 30
        }
        resp = self.client.post("/predict", json=incomplete_payload)
        self.assertEqual(resp.status_code, 422)

        # 2. Negative value for ge=0 constraint
        invalid_payload = {
            "pregnancies": 1,
            "glucose": -50.0,
            "blood_pressure": 70.0,
            "skin_thickness": 20.0,
            "insulin": 80.0,
            "bmi": 25.0,
            "dpf": 0.5,
            "age": 30
        }
        resp_invalid = self.client.post("/predict", json=invalid_payload)
        self.assertEqual(resp_invalid.status_code, 422)

    def test_zero_imputation_fallback(self):
        """Verify valid zero clinical values (missing test markers) are imputed using medians."""
        zero_payload = {
            "pregnancies": 0,
            "glucose": 0.0,        # Physiologically missing, imputed with median
            "blood_pressure": 0.0, # Physiologically missing, imputed with median
            "skin_thickness": 0.0, # Physiologically missing, imputed with median
            "insulin": 0.0,        # Physiologically missing, imputed with median
            "bmi": 0.0,            # Physiologically missing, imputed with median
            "dpf": 0.35,
            "age": 28
        }
        resp = self.client.post("/predict", json=zero_payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "success")
        self.assertIn("risk_probability", body)

    def test_model_insights_endpoint(self):
        """Verify /model-insights returns Gini feature importance for Random Forest."""
        resp = self.client.get("/model-insights")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model_type"], "Random Forest")
        self.assertIn("feature_importances", data)
        self.assertIn("top_risk_factors", data)

        # Sum of feature importances should be approximately 1.0 (100%)
        total_importance = sum(data["feature_importances"].values())
        self.assertAlmostEqual(total_importance, 1.0, places=2)

        # Verify descending order of top risk factors
        importances = [item["importance"] for item in data["top_risk_factors"]]
        self.assertEqual(importances, sorted(importances, reverse=True))

    def test_model_evaluation_endpoint(self):
        """Verify /model-evaluation contains comparison table and RF metrics."""
        resp = self.client.get("/model-evaluation")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["deployed_model"], "Random Forest")
        self.assertIn("comparison_table", data)
        self.assertIn("confusion_matrix", data)
        self.assertIn("roc_curve", data)

        # Verify confusion matrix keys and non-negative counts
        cm = data["confusion_matrix"]
        for key in ["tn", "fp", "fn", "tp"]:
            self.assertIn(key, cm)
            self.assertGreaterEqual(cm[key], 0)
        self.assertEqual(cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"], 154)

        # Verify ROC AUC
        roc = data["roc_curve"]
        self.assertGreater(roc["auc"], 0.70)
        self.assertEqual(len(roc["fpr"]), len(roc["tpr"]))

        # Verify models in comparison table
        model_names = [row["Model"] for row in data["comparison_table"]]
        self.assertIn("Logistic Regression", model_names)
        self.assertIn("Random Forest", model_names)


if __name__ == "__main__":
    unittest.main()
