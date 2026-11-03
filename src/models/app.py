# src/models/app.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="INX Future Inc. | Performance Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    artifact_path = os.path.join(
        base_dir, "models", "inx_performance_artifacts.joblib"
    )

    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Artifact not found at {artifact_path}. Please run export_pipeline.py first."
        )

    return joblib.load(artifact_path)


try:
    artifacts = load_artifacts()
    rf_model = artifacts["rf_model"]
    xgb_model = artifacts["xgb_model"]
    scaler = artifacts["scaler"]
    numerical_columns = artifacts["numerical_columns"]
    feature_names = artifacts["feature_names"]
    weights = artifacts.get(
        "weights", artifacts.get("optimal_weights", {"rf": 0.9, "xgb": 0.1})
    )
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# --- SIDEBAR: GUIDELINES & CONTEXT ---
with st.sidebar:
    st.title(" User Guide & Context")
    st.markdown(
        """
    ### **Business Objective**
    Address service delivery escalations and protect INX's top-tier employer brand by accurately identifying performance drivers and forecasting ratings.

    ---
    ### **Performance Tiers**
    *  **Tier 2 (Underperformer):** Falling short of delivery benchmarks; priority candidate for root-cause diagnosis and skill coaching.
    *  **Tier 3 (Standard):** Baseline contributor meeting operational expectations (accounts for ~73% of the workforce).
    *  **Tier 4 (High Performer):** Critical asset exceeding targets; candidate for fast-track leadership and retention programs.

    ---
    ### **How to Use**
    1. Adjust employee parameters across Demographics, Compensation, and Tenure.
    2. Click **Run Performance Evaluation**.
    3. Review the predicted tier, ensemble confidence breakdown, and actionable HR recommendations below.
    """
    )

# --- MAIN PAGE HEADER ---
st.title(" INX Future Inc. — Employee Performance Decision Engine")
st.caption(
    "Powered by an Optimized Soft-Voting Ensemble (90% Tuned Random Forest + 10% XGBoost | Test Accuracy: 94.0%)"
)

# Expandable Instruction Box for first-time visitors
with st.expander(" Click here to view how the scoring model works"):
    st.write(
        """
    This tool evaluates multidimensional HR signals — including work environment sentiment, career velocity, compensation trajectory, and job roles — to forecast performance ratings.
    Predictions are intended as an objective, evidence-based triage system to support HR business partners and department heads, **not** as automated punitive actions.
    """
    )

st.markdown("---")

# --- INPUT FORM ---
with st.form("employee_input_form"):
    st.subheader("1. Employee Profile Input")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics & Commute**")
        age = st.number_input("Age", min_value=18, max_value=65, value=36)
        gender = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox(
            "Marital Status", ["Married", "Single", "Divorced"]
        )
        distance = st.number_input(
            "Distance From Home (km)", min_value=1, max_value=50, value=9
        )
        travel = st.selectbox(
            "Business Travel Frequency",
            ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
        )
        education_bg = st.selectbox(
            "Education Background",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Human Resources",
                "Other",
            ],
        )
        education_lvl = st.select_slider(
            "Education Level",
            options=[1, 2, 3, 4, 5],
            value=3,
            help="1: Below College | 2: College | 3: Bachelor | 4: Master | 5: Doctor",
        )

    with col2:
        st.markdown("**Role, Level & Compensation**")
        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Developer",
                "Manager R&D",
                "Research Scientist",
                "Sales Representative",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Data Scientist",
                "Business Analyst",
                "Delivery Manager",
                "Technical Architect",
                "Finance Manager",
                "Human Resources",
                "Manager",
            ],
        )
        job_level = st.slider("Job Level (1 = Entry, 5 = Executive)", 1, 5, 2)
        hourly_rate = st.number_input(
            "Hourly Rate ($)", min_value=30, max_value=100, value=66
        )
        salary_hike = st.slider(
            "Last Salary Hike (%)",
            min_value=10,
            max_value=30,
            value=15,
            help="Significant driver of employee motivation.",
        )
        companies_worked = st.number_input(
            "Num Companies Worked", min_value=0, max_value=15, value=2
        )
        trainings = st.slider("Trainings Attended Last Year", 0, 10, 3)

    with col3:
        st.markdown("**Tenure & Workplace Sentiment**")
        total_exp = st.number_input("Total Experience (Years)", 0, 40, 11)
        years_company = st.number_input("Tenure at Company (Years)", 0, 40, 7)
        years_in_role = st.number_input("Years in Current Role", 0, 30, 4)
        years_promotion = st.number_input(
            "Years Since Last Promotion", 0, 20, 2
        )
        years_manager = st.number_input(
            "Years With Current Manager", 0, 20, 3
        )

        st.caption("Sentiment Ratings (1 = Low, 4 = Very High)")
        env_satisfaction = st.slider(
            "Environment Satisfaction", 1, 4, 3, help="Strongest model driver."
        )
        job_involvement = st.slider("Job Involvement", 1, 4, 3)
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        rel_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
        work_life = st.slider("Work-Life Balance", 1, 4, 3)

    submit = st.form_submit_button(
        " Run Performance Evaluation", use_container_width=True, type="primary"
    )

# --- RESULTS SECTION ---
if submit:
    # 1. Base input dictionary
    travel_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    input_data = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "DistanceFromHome": distance,
        "EmpHourlyRate": hourly_rate,
        "NumCompaniesWorked": companies_worked,
        "EmpLastSalaryHikePercent": salary_hike,
        "TotalWorkExperienceInYears": total_exp,
        "TrainingTimesLastYear": trainings,
        "ExperienceYearsAtThisCompany": years_company,
        "ExperienceYearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": years_promotion,
        "YearsWithCurrManager": years_manager,
        "EmpEducationLevel": education_lvl,
        "EmpEnvironmentSatisfaction": env_satisfaction,
        "EmpJobInvolvement": job_involvement,
        "EmpJobLevel": job_level,
        "EmpJobSatisfaction": job_satisfaction,
        "EmpRelationshipSatisfaction": rel_satisfaction,
        "EmpWorkLifeBalance": work_life,
        "BusinessTravelFrequency": travel_map[travel],
    }

    input_df = pd.DataFrame([input_data])

    # 2. Build one-hot encoded columns matching training schema
    for feat in feature_names:
        if feat.startswith("EducationBackground_"):
            val = feat.replace("EducationBackground_", "")
            input_df[feat] = 1 if education_bg == val else 0
        elif feat.startswith("EmpJobRole_"):
            val = feat.replace("EmpJobRole_", "")
            input_df[feat] = 1 if job_role == val else 0
        elif feat.startswith("MaritalStatus_"):
            val = feat.replace("MaritalStatus_", "")
            input_df[feat] = 1 if marital_status == val else 0
        elif feat not in input_df.columns:
            input_df[feat] = 0

    input_df = input_df[feature_names]

    # 3. Standardize numerical inputs
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # 4. Predict probabilities with Tuned Ensemble
    rf_probs = rf_model.predict_proba(input_df)
    xgb_probs = xgb_model.predict_proba(input_df)

    w_rf = weights.get("rf", 0.9)
    w_xgb = weights.get("xgb", 0.1)
    blended_probs = (w_rf * rf_probs) + (w_xgb * xgb_probs)[0]
    predicted_tier = int(np.argmax(blended_probs, axis=1)[0] + 2)
    confidence = float(np.max(blended_probs) * 100)

    st.markdown("---")
    st.subheader("2. Evaluation Outcome")

    res_col1, res_col2 = st.columns([1, 1.5])

    with res_col1:
        if predicted_tier == 2:
            st.error(
                f"### Predicted: Tier 2 (Underperformer)\n**Confidence: {confidence:.1f}%**"
            )
            st.write(
                " **Operational Risk Flag:** Profile exhibits characteristics associated with high service delivery escalation and engagement decline."
            )
        elif predicted_tier == 3:
            st.info(
                f"### Predicted: Tier 3 (Standard Performer)\n**Confidence: {confidence:.1f}%**"
            )
            st.write(
                " **Stable Contributor:** Profile reflects core operational competency and steady role execution."
            )
        else:
            st.success(
                f"### Predicted: Tier 4 (High Performer)\n**Confidence: {confidence:.1f}%**"
            )
            st.write(
                " **High-Value Talent:** Profile exhibits high output and top-tier organizational contribution."
            )

    with res_col2:
        st.markdown("**Probability Distribution Across Performance Tiers**")
        prob_chart_data = pd.DataFrame(
            {
                "Tier": [
                    "Tier 2 (Underperformer)",
                    "Tier 3 (Standard)",
                    "Tier 4 (High Performer)",
                ],
                "Probability (%)": blended_probs[0] * 100,
            }
        ).set_index("Tier")
        st.bar_chart(prob_chart_data)

    # --- STRATEGIC RECOMMENDATIONS (ALWAYS VISIBLE & ACTIONABLE) ---
    st.markdown("---")
    st.subheader("3. Strategic HR & Management Recommendations")

    # Tier-specific overarching guidance
    if predicted_tier == 2:
        st.error(
            """
        **Priority Action Plan for Tier 2 Profile:**
        * **Root-Cause Diagnostic:** Initiate an empathetic, non-punitive 1-on-1 discovery session to uncover tooling blockers, process friction, or burnout.
        * **Targeted Upskilling Plan:** Pair the employee with a senior mentor and enroll in focused technical or delivery training.
        * **Quarterly Milestone Tracking:** Establish clear, measurable 30-60-90 day performance goals before initiating formal PIP pathways.
        """
        )
    elif predicted_tier == 3:
        st.info(
            """
        **Growth & Progression Plan for Tier 3 Profile:**
        * **Role Expansion:** Offer opportunities to take ownership of cross-departmental tasks or client-facing deliverables.
        * **Promotion Readiness Check:** Assess whether the employee is approaching career stagnation (especially if tenure in role > 3 years).
        * **Engagement Check-ins:** Maintain regular touchpoints on workplace sentiment to prevent slip into Tier 2.
        """
        )
    else:
        st.success(
            """
        **Retention & Leadership Plan for Tier 4 Profile:**
        * **Fast-Track Leadership:** Identify candidates for mentorship, technical leadership, or project management tracks.
        * **Competitive Merit Adjustment:** Ensure salary hike and equity/bonus incentives reflect top-tier market benchmarks to prevent poaching.
        * **Autonomy & Empowerment:** Grant autonomy over high-impact strategic initiatives to maintain intrinsic engagement.
        """
        )

    # Granular Risk Indicators
    st.markdown("##### Specific Factor Observations for this Record:")
    alerts = []
    if env_satisfaction <= 2:
        alerts.append(
            " **Low Environment Satisfaction:** This is the single strongest statistical driver of performance deterioration ($\beta = 1.23$). Investigate team culture, manager rapport, or working tools immediately."
        )
    if years_promotion >= 3 and years_in_role >= 4:
        alerts.append(
            " **Career Stagnation Alert:** Over 3 years without a promotion combined with long role tenure creates a strong downward pull on employee engagement."
        )
    if salary_hike < 13:
        alerts.append(
            " **Below-Average Compensation Growth:** Recent salary hikes below 13% correlate with reduced delivery velocity and lower performance scores."
        )
    if work_life <= 2:
        alerts.append(
            " **Work-Life Balance Deficit:** Low work-life sentiment indicates elevated risk of burnout."
        )

    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.caption(
            " No critical negative risk indicators detected across compensation, environment sentiment, or promotion tenure."
        )