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



# ==========================================
# PAGE 1: RISK PREDICTOR
# ==========================================
if page == "Risk Predictor":
    st.title("🏥 Diabetes Health Risk Predictor")
    st.markdown("""
    Provide a patient's clinical measurements below to evaluate their risk of diabetes. 
    This page uses the deployed **Logistic Regression** model for instant classification, confidence assessment, 
    and feature-level risk contribution analysis.
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
            help="Plasma glucose concentration at 2 hours in an oral glucose tolerance test. Normal range is < 140 mg/dL."
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
            help="2-hour serum insulin. Normal fasting is under 25."
        )
        
        bmi = st.number_input(
            "Body Mass Index (BMI)", 
            min_value=0.0, 
            max_value=100.0, 
            value=25.0,
            format="%.1f",
            help="Normal weight range is 18.5 - 24.9. Obese is >= 30."
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
                response = requests.post(f"{API_URL}/predict", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    prediction = result["prediction"]
                    probability = result["risk_probability"]
                    confidence_level = result["confidence_level"]
                    contributions = result["contributions"]
                    
                    # Layout for Results
                    res_col1, res_col2 = st.columns([1, 1])
                    
                    with res_col1:
                        st.subheader("Model Diagnostic Outcome")
                        
                        # Set color alerts based on outcome
                        if prediction == 1:
                            st.error(f"⚠️ **High Risk of Diabetes Detected**\n\nThe model predicts a high risk of diabetes.")
                        else:
                            st.success(f"✅ **Low Risk of Diabetes Detected**\n\nThe model predicts a low risk of diabetes.")
                            
                        # Confidence level card
                        st.metric(
                            label="Model Confidence Level", 
                            value=confidence_level, 
                            delta=f"Probability: {probability * 100:.1f}%" if prediction == 1 else f"Probability: {(1 - probability) * 100:.1f}%",
                            delta_color="off"
                        )
                        
                        # Progress bar of risk
                        st.write("**Risk Probability Meter:**")
                        st.progress(probability)
                        st.caption(f"Raw risk probability: {probability:.4f} (Decision threshold: 0.5000)")
                        
                    with res_col2:
                        st.subheader("Clinical Factors Contribution")
                        st.markdown("""
                        This chart explains *why* the model made this prediction for this patient.
                        - **Positive values (right):** Factors that **increased** the diabetes risk.
                        - **Negative values (left):** Factors that **reduced** the diabetes risk.
                        """)
                        
                        contrib_df = pd.DataFrame(
                            list(contributions.items()), 
                            columns=["Clinical Metric", "Log-Odds Contribution"]
                        ).sort_values(by="Log-Odds Contribution")
                        
                        # Display contribution bar chart
                        st.bar_chart(
                            data=contrib_df,
                            x="Clinical Metric",
                            y="Log-Odds Contribution",
                            horizontal=True
                        )
                        
                        # Highlight key driver
                        max_factor = contrib_df.loc[contrib_df["Log-Odds Contribution"].idxmax()]
                        if max_factor["Log-Odds Contribution"] > 0:
                            st.warning(f"💡 **Key Driver:** **{max_factor['Clinical Metric']}** was the primary factor driving the risk score up.")
                    
                    # ----------------------------------------
                    # Lifestyle Recommendation Engine
                    # ----------------------------------------
                    st.markdown("---")
                    st.subheader("📋 Personalized Recommendations")
                    
                    recs = []
                    
                    if bmi >= 30.0:
                        recs.append("🏋️ **Weight Management:** Your BMI is in the clinical obesity range (>= 30). We recommend consulting a nutritionist/dietitian and introducing moderate-intensity cardiovascular exercise (such as brisk walking) for at least 150 minutes per week to help manage visceral fat and weight.")
                    elif bmi >= 25.0:
                        recs.append("🥗 **Healthy Weight Advice:** Your BMI is in the overweight range (25.0 - 29.9). Balanced portion sizes and reducing processed sugar consumption can help prevent progression to higher risk categories.")
                        
                    if glucose >= 140.0:
                        recs.append("🩺 **Consult a Physician:** Your glucose level is elevated (>= 140 mg/dL). We highly recommend consulting a primary healthcare provider or endocrinologist for a diagnostic HbA1c test or an oral glucose tolerance test (OGTT).")
                    elif glucose < 70.0 and glucose > 0:
                        recs.append("🍬 **Hypoglycemia Alert:** Your glucose level is quite low (< 70 mg/dL). Monitor for symptoms like shakiness, sweating, or dizziness, and discuss these readings with a medical professional.")
                        
                    if age >= 45:
                        recs.append("📋 **Regular Screening Protocol:** American Diabetes Association (ADA) guidelines suggest that adults aged 45 and older should undergo regular diabetes screening every 3 years, regardless of symptoms.")
                        
                    if blood_pressure >= 130.0:
                        recs.append("❤️ **Hypertension Monitoring:** Your diastolic blood pressure is elevated (>= 80 mm Hg / Stage 1 or 2 Hypertension ranges). Reducing dietary sodium, managing daily stressors, and tracking blood pressure at home are recommended cardiovascular safeguards.")
                        
                    if insulin >= 166.0:
                        recs.append("🧬 **Insulin Sensitivity Review:** Your insulin levels are high, which can be an early clinical indicator of insulin resistance. Incorporating weight lifting, resistance exercises, and high-fiber complex carbohydrates can improve insulin sensitivity.")
                        
                    if not recs:
                        st.success("🎉 **Keep up the good work!** All entered clinical metrics (BMI, Glucose, BP, Age, Insulin) currently fall within standard healthy ranges. Maintain a balanced diet and active lifestyle.")
                    else:
                        for rec in recs:
                            st.info(rec)
                            

                    
                else:
                    st.error("Failed to connect to the FastAPI server. Please check the backend log.")
            except Exception as e:
                st.error(f"Connection error to backend API: {e}")
                st.info("Ensure the FastAPI server is running locally on port 8001.")

# ==========================================
# PAGE 2: MODEL INSIGHTS
# ==========================================
elif page == "Model Insights":
    st.title("📊 Global Model Insights")
    st.markdown("""
    This page shows the global characteristics of our trained **Logistic Regression** model. Unlike local patient explanations, 
    this shows which variables the model has learned are generally the most important indicators of diabetes across the whole dataset.
    """)
    
    with st.spinner("Fetching global insights from FastAPI backend..."):
        try:
            response = requests.get(f"{API_URL}/model-insights")
            if response.status_code == 200:
                insights = response.json()
                
                coefs = insights["coefficients"]
                top_factors = insights["top_risk_factors"]
                intercept = insights["intercept"]
                
                # Visual layout for coefficients
                col_chart, col_list = st.columns([1.2, 1.0])
                
                with col_chart:
                    st.subheader("Global Logistic Regression Coefficients")
                    st.markdown("""
                    The coefficient weights indicate the effect of changing each standardized feature by 1 standard deviation:
                    - **Positive values (Red/Orange):** Increase the overall probability of a diabetes prediction.
                    - **Negative values (Green):** Decrease the overall probability of a diabetes prediction.
                    """)
                    
                    insights_df = pd.DataFrame(
                        list(coefs.items()), 
                        columns=["Clinical Metric", "Coefficient Weight"]
                    ).sort_values(by="Coefficient Weight")
                    
                    st.bar_chart(
                        data=insights_df,
                        x="Clinical Metric",
                        y="Coefficient Weight",
                        horizontal=True
                    )
                    st.caption(f"Model Intercept (Base Bias): {intercept:.4f}")
                    
                with col_list:
                    st.subheader("Top Risk Factors (Ranked by Influence)")
                    st.markdown("Features are ranked by their **absolute weight**, representing their total relative contribution strength in either direction:")
                    
                    # Find maximum absolute coefficient to scale the progress bars
                    max_abs_coef = max([f["abs_coefficient"] for f in top_factors])
                    
                    # Render ranked progress bars
                    for rank, factor in enumerate(top_factors, 1):
                        feature_name = factor["feature"]
                        coefficient = factor["coefficient"]
                        abs_coef = factor["abs_coefficient"]
                        
                        # Calculate percentage for progress bar
                        scaled_val = float(abs_coef / max_abs_coef)
                        
                        # Build layout
                        st.write(f"**{rank}. {feature_name}**")
                        st.progress(scaled_val)
                        
                        if coefficient > 0:
                            st.caption(f"Influence Score: {coefficient:+.4f} (Increases Risk)")
                        else:
                            st.caption(f"Influence Score: {coefficient:+.4f} (Reduces Risk)")
                            
            else:
                st.error("Failed to load insights from backend.")
        except Exception as e:
            st.error(f"Error connecting to backend API: {e}")

# ==========================================
# PAGE 3: MODEL EVALUATION
# ==========================================
elif page == "Model Evaluation":
    st.title("🧪 Model Training & Performance Evaluation")
    st.markdown("""
    In compliance with production-grade engineering, we trained multiple model architectures on the split dataset 
    to compare performance. Review the evaluation metrics below.
    """)
    
    with st.spinner("Fetching evaluation metrics from FastAPI backend..."):
        try:
            response = requests.get(f"{API_URL}/model-evaluation")
            if response.status_code == 200:
                eval_data = response.json()
                
                comp_table = eval_data["comparison_table"]
                cm = eval_data["confusion_matrix"]
                roc = eval_data["roc_curve"]
                
                # 1. Comparison Table
                st.subheader("Model Performance Comparison")
                st.markdown("Metrics computed on the hold-out testing partition (20% of the dataset):")
                
                comp_df = pd.DataFrame(comp_table)
                st.dataframe(comp_df, use_container_width=True)
                
                st.markdown("---")
                
                # Plots layout
                plot_col1, plot_col2 = st.columns(2)
                
                # 2. Confusion Matrix Plot
                with plot_col1:
                    st.subheader("Confusion Matrix (Deployed Model)")
                    
                    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    
                    # Modern styling
                    matrix_data = np.array([[tn, fp], [fn, tp]])
                    im = ax.imshow(matrix_data, cmap="Blues", interpolation="nearest")
                    
                    # Annotations
                    labels = [
                        [f"True Neg (TN)\n{tn}", f"False Pos (FP)\n{fp}"],
                        [f"False Neg (FN)\n{fn}", f"True Pos (TP)\n{tp}"]
                    ]
                    
                    for i in range(2):
                        for j in range(2):
                            text_color = "white" if matrix_data[i, j] > (matrix_data.max() / 2) else "black"
                            ax.text(j, i, labels[i][j], ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")
                            
                    # Labels configuration
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
                    - **True Negatives (TN):** Healthy patients predicted correctly as low risk.
                    - **False Positives (FP):** Healthy patients incorrectly flagged as high risk (causes false alarm/anxiety).
                    - **False Negatives (FN):** Patients with diabetes who were missed by the model.
                    - **True Positives (TP):** Patients with diabetes correctly identified.
                    
                    *In a clinical setting, **False Negatives are the most critical metric**. Missing a diabetic patient (FN) is far more dangerous than flagging a healthy patient for follow-up testing (FP). Thus, clinicians focus on maximizing **Recall** (Sensitivity) to minimize False Negatives.*
                    """)
                    
                # 3. ROC Curve Plot
                with plot_col2:
                    st.subheader("ROC Curve (Deployed Model)")
                    
                    fpr, tpr, auc_val = roc["fpr"], roc["tpr"], roc["auc"]
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    
                    # Plot ROC line
                    ax.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"Logistic Regression (AUC = {auc_val:.4f})")
                    # Plot diagonal reference
                    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Guess (AUC = 0.50)")
                    
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
                    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
                    ax.set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
                    ax.legend(loc="lower right")
                    ax.grid(True, linestyle=":", alpha=0.6)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    st.markdown(f"""
                    **Understanding the ROC Curve & AUC Score:**
                    - **ROC Curve:** Plots the True Positive Rate (Recall) against the False Positive Rate at all possible classification thresholds.
                    - **AUC Score (Area Under Curve):** Quantifies the overall ability of the model to distinguish between classes. 
                    - An **AUC of {auc_val:.4f}** indicates that if you randomly pick one patient with diabetes and one patient without, the model has an **{auc_val*100:.1f}% chance** of assigning a higher risk probability to the diabetic patient.
                    """)
                    
            else:
                st.error("Failed to load evaluation data from backend.")
        except Exception as e:
            st.error(f"Error connecting to backend API: {e}")
