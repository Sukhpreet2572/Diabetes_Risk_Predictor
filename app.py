import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page configurations
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Backend URL config
API_URL = "http://127.0.0.1:8001"

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/150/hospital-room.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to Page:", ["Risk Predictor", "Model Insights", "Model Evaluation"])

# Sidebar model badge
st.sidebar.markdown("---")
st.sidebar.markdown("**Deployed Architecture:**")
st.sidebar.info("🌲 **Random Forest Classifier**\n- 100 Estimators\n- Stratified 80/20 Split\n- Leakage-Free Pipeline")


# ==========================================
# PAGE 1: RISK PREDICTOR
# ==========================================
if page == "Risk Predictor":
    st.title("🏥 Diabetes Health Risk Predictor")
    st.markdown("""
    Provide a patient's clinical measurements below to evaluate their risk of diabetes. 
    This page uses the deployed **Random Forest** ensemble model for instant classification, confidence assessment, 
    and patient-level feature attribution.
    """)
    
    # Create clean inputs using columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vital Statistics")
        pregnancies = st.slider(
            "Pregnancies", 
            min_value=0, 
            max_value=20, 
            value=1,
            help="Number of times pregnant"
        )
        
        glucose = st.number_input(
            "Glucose Level (mg/dL)", 
            min_value=0.0, 
            max_value=500.0, 
            value=120.0,
            help="Plasma glucose concentration at 2 hours in an oral glucose tolerance test. Normal fasting range is < 100 mg/dL; < 140 mg/dL normal post-meal."
        )
        
        blood_pressure = st.number_input(
            "Diastolic Blood Pressure (mm Hg)", 
            min_value=0.0, 
            max_value=250.0, 
            value=70.0,
            help="Diastolic blood pressure. Normal range is < 80 mm Hg."
        )
        
        skin_thickness = st.slider(
            "Triceps Skin Fold Thickness (mm)", 
            min_value=0, 
            max_value=120, 
            value=20,
            help="Used to estimate body fat. Normal range is typically 10-30 mm."
        )
    
    with col2:
        st.subheader("Body Metrics")
        insulin = st.number_input(
            "Insulin Level (mu U/ml)", 
            min_value=0.0, 
            max_value=1200.0, 
            value=80.0,
            help="2-hour serum insulin. Normal fasting is under 25 mu U/ml."
        )
        
        bmi = st.number_input(
            "Body Mass Index (BMI)", 
            min_value=0.0, 
            max_value=100.0, 
            value=25.0,
            format="%.1f",
            help="Normal weight range is 18.5 - 24.9. Overweight is 25 - 29.9. Obese is >= 30."
        )
        
        dpf = st.number_input(
            "Diabetes Pedigree Function (Hereditary Score)", 
            min_value=0.0, 
            max_value=3.0, 
            value=0.500,
            format="%.3f",
            help="A score representing genetic/heredity history of diabetes. Higher value means stronger family history."
        )
        
        age = st.slider(
            "Age", 
            min_value=1, 
            max_value=120, 
            value=33,
            help="Age of the patient in years"
        )
        
    st.markdown("---")
    
    # ----------------------------
    # Clinical Validation Engine
    # ----------------------------
    validation_warnings = []
    
    # Check clinical rules
    if age < 18:
        validation_warnings.append(f"Patient age is {age} (pediatric). The Pima Indians dataset represents adult females (>= 21 years).")
    if bmi > 70.0:
        validation_warnings.append(f"BMI of {bmi} is extremely high (typical maximum clinical index is 70).")
    if glucose > 400.0:
        validation_warnings.append(f"Glucose level of {glucose} mg/dL is extremely high (normal fasting range is < 100 mg/dL).")
    if insulin > 900.0:
        validation_warnings.append(f"Insulin level of {insulin} mu U/ml is outside typical clinical boundaries.")
    if blood_pressure > 180.0:
        validation_warnings.append(f"Diastolic blood pressure of {blood_pressure} mm Hg indicates extreme crisis/outlier range.")
    if blood_pressure < 30.0 and blood_pressure > 0:
        validation_warnings.append(f"Diastolic blood pressure of {blood_pressure} mm Hg is extremely low.")
        
    # Render warnings if present
    if validation_warnings:
        st.warning("⚠️ **Clinical Outliers Detected:**\n" + "\n".join([f"- {w}" for w in validation_warnings]) + "\n\n*The model was not trained on these extreme values. You may still run the prediction, but results could be unreliable.*")

    # Prediction trigger
    if st.button("📊 Assess Diabetes Risk", use_container_width=True):
        payload = {
            "pregnancies": pregnancies,
            "glucose": glucose,
            "blood_pressure": blood_pressure,
            "skin_thickness": skin_thickness,
            "insulin": insulin,
            "bmi": bmi,
            "dpf": dpf,
            "age": age
        }
        
        with st.spinner("Executing backend model inference..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    prediction = result["prediction"]
                    probability = result["risk_probability"]
                    confidence_level = result["confidence_level"]
                    contributions = result.get("contributions", {})
                    base_prior = result.get("intercept", None)
                    
                    # Layout for Results
                    res_col1, res_col2 = st.columns([1, 1])
                    
                    with res_col1:
                        st.subheader("Model Diagnostic Outcome")
                        
                        # Set color alerts based on outcome
                        if prediction == 1:
                            st.error("⚠️ **High Risk of Diabetes Detected**\n\nThe Random Forest model predicts an elevated likelihood of diabetes.")
                        else:
                            st.success("✅ **Low Risk of Diabetes Detected**\n\nThe Random Forest model predicts a low likelihood of diabetes.")
                            
                        # Confidence level card
                        st.metric(
                            label="Model Confidence Level", 
                            value=confidence_level, 
                            delta=f"Risk Probability: {probability * 100:.1f}%",
                            delta_color="off"
                        )
                        
                        # Progress bar of risk
                        st.write("**Risk Probability Meter:**")
                        st.progress(min(max(probability, 0.0), 1.0))
                        st.caption(f"Predicted risk probability: **{probability:.4f}** (Decision threshold: 0.5000)")
                        if base_prior is not None:
                            st.caption(f"Forest Population Base Rate: **{base_prior:.4f}**")
                        
                    with res_col2:
                        st.subheader("Patient Clinical Factor Attribution")
                        st.markdown("""
                        **Local Prediction Explanation (Decision-Path Attribution):**
                        - **Positive values (right):** Factors that **increased** this patient's risk probability above baseline.
                        - **Negative values (left):** Factors that **reduced** this patient's risk probability below baseline.
                        """)
                        
                        contrib_df = pd.DataFrame(
                            list(contributions.items()), 
                            columns=["Clinical Metric", "Risk Contribution"]
                        ).sort_values(by="Risk Contribution")
                        
                        # Display contribution bar chart
                        st.bar_chart(
                            data=contrib_df,
                            x="Clinical Metric",
                            y="Risk Contribution",
                            horizontal=True
                        )
                        
                        # Highlight key driver
                        if not contrib_df.empty:
                            max_factor = contrib_df.loc[contrib_df["Risk Contribution"].idxmax()]
                            if max_factor["Risk Contribution"] > 0:
                                st.warning(f"💡 **Key Risk Driver:** **{max_factor['Clinical Metric']}** (+{max_factor['Risk Contribution']:.4f}) was the primary factor driving risk upward.")
                            else:
                                st.info("💡 **Protective Factors:** All individual measurements contributed neutrally or protectively relative to baseline.")
                    
                    # ----------------------------------------
                    # Lifestyle Recommendation Engine
                    # ----------------------------------------
                    st.markdown("---")
                    st.subheader("📋 Personalized Clinical & Lifestyle Recommendations")
                    
                    recs = []
                    
                    if bmi >= 30.0:
                        recs.append("🏋️ **Weight Management:** BMI is in the clinical obesity range (>= 30). Recommend consulting a registered dietitian and initiating 150+ minutes/week of moderate cardiovascular activity.")
                    elif bmi >= 25.0:
                        recs.append("🥗 **Healthy Weight Advice:** BMI is in the overweight range (25.0 - 29.9). Balanced portion control and minimizing refined carbohydrates can prevent progression.")
                        
                    if glucose >= 140.0:
                        recs.append("🩺 **Elevated Glucose Evaluation:** Post-challenge glucose is elevated (>= 140 mg/dL). Strongly recommend a formal diagnostic fasting plasma glucose or HbA1c test with a physician.")
                    elif glucose < 70.0 and glucose > 0:
                        recs.append("🍬 **Hypoglycemia Alert:** Blood glucose reading is below normal physiological fasting levels (< 70 mg/dL). Investigate potential hypoglycemic episodes.")
                        
                    if age >= 45:
                        recs.append("📋 **Standard Screening Protocol:** Per American Diabetes Association (ADA) guidelines, adults aged 45 and older should undergo routine diabetes screening every 3 years.")
                        
                    if blood_pressure >= 130.0:
                        recs.append("❤️ **Blood Pressure Review:** Diastolic pressure is elevated. Recommend home blood pressure tracking and reduction in dietary sodium.")
                        
                    if insulin >= 166.0:
                        recs.append("🧬 **Insulin Sensitivity Assessment:** Elevated insulin may indicate underlying insulin resistance. Resistance training and high-fiber diets can help improve insulin sensitivity.")
                        
                    if not recs:
                        st.success("🎉 **All Entered Metrics are Within Healthy Ranges:** Maintain current balanced nutrition and regular physical activity.")
                    else:
                        for rec in recs:
                            st.info(rec)
                            
                else:
                    st.error(f"Backend API error (Status Code: {response.status_code}). Detail: {response.text}")
            except Exception as e:
                st.error(f"Unable to connect to backend API at {API_URL}: {e}")
                st.info("Please verify that the FastAPI backend server is running via `python main.py` on port 8001.")

# ==========================================
# PAGE 2: MODEL INSIGHTS
# ==========================================
elif page == "Model Insights":
    st.title("📊 Global Model Insights")
    st.markdown("""
    This page presents the global feature importance learned by our deployed **Random Forest** ensemble. 
    Unlike the local patient-level attribution on the *Risk Predictor* page, global importance reflects 
    which clinical variables are most influential overall across the entire training population.
    """)
    
    with st.spinner("Fetching global model insights from FastAPI backend..."):
        try:
            response = requests.get(f"{API_URL}/model-insights", timeout=10)
            if response.status_code == 200:
                insights = response.json()
                
                feature_imp = insights.get("feature_importances", insights.get("coefficients", {}))
                top_factors = insights.get("top_risk_factors", [])
                model_type = insights.get("model_type", "Random Forest")
                
                # Visual layout
                col_chart, col_list = st.columns([1.2, 1.0])
                
                with col_chart:
                    st.subheader(f"Global {model_type} Feature Importance (Gini / MDI)")
                    st.markdown("""
                    Features are measured by **Mean Decrease in Impurity (MDI / Gini Importance)** across all 100 decision trees:
                    - Values sum to **1.0 (100%)**.
                    - A higher score indicates a feature that creates larger, cleaner splits throughout the ensemble.
                    """)
                    
                    insights_df = pd.DataFrame(
                        list(feature_imp.items()), 
                        columns=["Clinical Metric", "Importance Score"]
                    ).sort_values(by="Importance Score")
                    
                    st.bar_chart(
                        data=insights_df,
                        x="Clinical Metric",
                        y="Importance Score",
                        horizontal=True
                    )
                    
                with col_list:
                    st.subheader("Top Risk Factors (Ranked by Ensemble Importance)")
                    st.markdown("Relative ranking across all trained decision trees:")
                    
                    max_imp = max([f.get("importance", f.get("abs_coefficient", 0.0)) for f in top_factors]) if top_factors else 1.0
                    
                    for rank, factor in enumerate(top_factors, 1):
                        feature_name = factor["feature"]
                        imp = factor.get("importance", factor.get("abs_coefficient", 0.0))
                        
                        scaled_val = float(imp / max_imp) if max_imp > 0 else 0.0
                        
                        st.write(f"**{rank}. {feature_name}**")
                        st.progress(min(max(scaled_val, 0.0), 1.0))
                        st.caption(f"Importance: **{imp:.4f}** ({imp * 100:.1f}% relative share)")
                        
                st.markdown("---")
                st.info("""
                🔍 **Global Feature Importance vs. Local Prediction Attribution:**
                - **Global Importance (This Page):** Summarizes the average utility of each feature across every split in all 100 decision trees. It tells you that **Glucose** and **BMI** are generally the most decisive features in the dataset.
                - **Local Attribution (Risk Predictor Page):** Shows the exact path a specific patient took through the ensemble. Even if a feature is globally important, for an individual with normal glucose and high blood pressure, the local driver will reflect their unique clinical profile.
                """)
                            
            else:
                st.error("Failed to load insights from backend.")
        except Exception as e:
            st.error(f"Error connecting to backend API: {e}")
            st.info("Ensure the FastAPI backend is running on port 8001 via `python main.py`.")

# ==========================================
# PAGE 3: MODEL EVALUATION
# ==========================================
elif page == "Model Evaluation":
    st.title("🧪 Model Training & Performance Evaluation")
    st.markdown("""
    To ensure evidence-based model selection, we trained and compared **Logistic Regression**, **Random Forest**, and **XGBoost** 
    on an identical stratified 80/20 train-test partition using leakage-free preprocessing.
    """)
    
    with st.spinner("Fetching evaluation metrics from FastAPI backend..."):
        try:
            response = requests.get(f"{API_URL}/model-evaluation", timeout=10)
            if response.status_code == 200:
                eval_data = response.json()
                
                comp_table = eval_data["comparison_table"]
                cm = eval_data["confusion_matrix"]
                roc = eval_data["roc_curve"]
                deployed_model = eval_data.get("deployed_model", "Random Forest")
                
                # 1. Comparison Table
                st.subheader("Model Performance Comparison (Holdout Test Set)")
                st.markdown("All metrics evaluated on the identical 20% holdout split (154 samples):")
                
                comp_df = pd.DataFrame(comp_table)
                st.dataframe(comp_df, use_container_width=True)
                
                st.success(f"🏆 **Deployed Architecture: {deployed_model}** — Achieved superior Accuracy, Precision, Recall, and ROC-AUC compared to baseline Logistic Regression.")
                
                st.markdown("---")
                
                # Plots layout
                plot_col1, plot_col2 = st.columns(2)
                
                # 2. Confusion Matrix Plot
                with plot_col1:
                    st.subheader(f"Confusion Matrix ({deployed_model})")
                    
                    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    
                    matrix_data = np.array([[tn, fp], [fn, tp]])
                    im = ax.imshow(matrix_data, cmap="Blues", interpolation="nearest")
                    
                    labels = [
                        [f"True Neg (TN)\n{tn}", f"False Pos (FP)\n{fp}"],
                        [f"False Neg (FN)\n{fn}", f"True Pos (TP)\n{tp}"]
                    ]
                    
                    for i in range(2):
                        for j in range(2):
                            text_color = "white" if matrix_data[i, j] > (matrix_data.max() / 2) else "black"
                            ax.text(j, i, labels[i][j], ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")
                            
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.set_xticklabels(["Low Risk (0)", "High Risk (1)"])
                    ax.set_yticklabels(["Low Risk (0)", "High Risk (1)"])
                    ax.set_xlabel("Predicted Label", fontweight="bold")
                    ax.set_ylabel("Actual Label", fontweight="bold")
                    fig.colorbar(im, ax=ax)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    st.markdown("""
                    **Understanding the Confusion Matrix:**
                    - **True Negatives (TN):** Healthy patients correctly classified as low risk.
                    - **False Positives (FP):** Healthy patients flagged as high risk (triggers follow-up confirmatory testing).
                    - **False Negatives (FN):** Diabetic patients missed by the model.
                    - **True Positives (TP):** Diabetic patients correctly identified.
                    
                    *In medical triage, minimizing **False Negatives** (maximizing Recall) is crucial to avoid missed diagnoses.*
                    """)
                    
                # 3. ROC Curve Plot
                with plot_col2:
                    st.subheader(f"ROC Curve ({deployed_model})")
                    
                    fpr, tpr, auc_val = roc["fpr"], roc["tpr"], roc["auc"]
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    
                    # Plot ROC line
                    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"{deployed_model} (AUC = {auc_val:.4f})")
                    # Plot diagonal reference
                    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
                    
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
                    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
                    ax.set_title(f"ROC Characteristic — {deployed_model}", fontweight="bold")
                    ax.legend(loc="lower right")
                    ax.grid(True, linestyle=":", alpha=0.6)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    st.markdown(f"""
                    **Understanding ROC-AUC:**
                    - **ROC-AUC ({auc_val:.4f}):** Indicates that for any randomly chosen pair of diabetic and non-diabetic patients, 
                      the model assigns a higher risk score to the diabetic patient **{auc_val * 100:.1f}%** of the time.
                    """)
                    
            else:
                st.error("Failed to load evaluation data from backend.")
        except Exception as e:
            st.error(f"Error connecting to backend API: {e}")
            st.info("Ensure the FastAPI backend is running on port 8001 via `python main.py`.")
