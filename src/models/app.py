# src/models/app.py
import io
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


# --- VECTORIZED BATCH PREDICTION PIPELINE ---
def process_and_predict_batch(input_df: pd.DataFrame):
    """Preprocesses arbitrary raw batch data matching training schema

    and computes ensemble probability distributions.
    """
    df_proc = input_df.copy()

    # Preserve identifiers if available
    emp_ids = (
        df_proc["EmpNumber"]
        if "EmpNumber" in df_proc.columns
        else pd.Series(range(1, len(df_proc) + 1))
    )

    # 1. Drop targets/redundant columns
    df_proc = df_proc.drop(
        columns=[
            "PerformanceRating",
            "EmpNumber",
            "Attrition",
            "OverTime",
            "EmpDepartment",
        ],
        errors="ignore",
    )

    # 2. Ordinal and binary conversions
    ordinal_cols = [
        "EmpEducationLevel",
        "EmpEnvironmentSatisfaction",
        "EmpJobInvolvement",
        "EmpJobLevel",
        "EmpJobSatisfaction",
        "EmpRelationshipSatisfaction",
        "EmpWorkLifeBalance",
    ]
    for col in ordinal_cols:
        if col in df_proc.columns:
            df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce").fillna(
                3
            )

    travel_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    if "BusinessTravelFrequency" in df_proc.columns:
        df_proc["BusinessTravelFrequency"] = (
            df_proc["BusinessTravelFrequency"].map(travel_map).fillna(1)
        )

    if "Gender" in df_proc.columns:
        df_proc["Gender"] = (
            df_proc["Gender"]
            .replace({"Female": 0, "Male": 1})
            .infer_objects(copy=False)
            .fillna(0)
            .astype(int)
        )

    # 3. Categorical one-hot dummies
    nominal_cols = ["EducationBackground", "EmpJobRole", "MaritalStatus"]
    present_nominals = [c for c in nominal_cols if c in df_proc.columns]
    if present_nominals:
        df_proc = pd.get_dummies(
            df_proc, columns=present_nominals, drop_first=True, dtype=int
        )

    # 4. Align with master feature schema
    for feat in feature_names:
        if feat not in df_proc.columns:
            df_proc[feat] = 0
    df_aligned = df_proc[feature_names].copy()

    # 5. Handle missing values & scale
    df_aligned[numerical_columns] = df_aligned[numerical_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    df_aligned = df_aligned.fillna(0)
    df_aligned[numerical_columns] = scaler.transform(
        df_aligned[numerical_columns]
    )

    # 6. Ensemble Predictions
    rf_probs = rf_model.predict_proba(df_aligned)
    xgb_probs = xgb_model.predict_proba(df_aligned)

    w_rf = weights.get("rf", 0.9)
    w_xgb = weights.get("xgb", 0.1)
    blended_probs = (w_rf * rf_probs) + (w_xgb * xgb_probs)

    predicted_tiers = np.argmax(blended_probs, axis=1) + 2
    max_confidence = np.max(blended_probs, axis=1) * 100

    # 7. Generate contextual Risk Flags per employee
    flags = []
    for idx, row in input_df.iterrows():
        emp_flags = []
        if row.get("EmpEnvironmentSatisfaction", 3) <= 2:
            emp_flags.append("Low Environment Satisfaction")
        if (
            row.get("YearsSinceLastPromotion", 0) >= 3
            and row.get("ExperienceYearsInCurrentRole", 0) >= 4
        ):
            emp_flags.append("Career Stagnation Risk")
        if row.get("EmpLastSalaryHikePercent", 15) < 13:
            emp_flags.append("Below-Average Salary Hike")
        if row.get("EmpWorkLifeBalance", 3) <= 2:
            emp_flags.append("Burnout / Work-Life Risk")

        flags.append("; ".join(emp_flags) if emp_flags else "No Critical Flags")

    # Output DataFrame
    results_df = input_df.copy()
    results_df.insert(0, "Predicted_PerformanceRating", predicted_tiers)
    results_df.insert(1, "Model_Confidence_Pct", max_confidence.round(1))
    results_df.insert(2, "Operational_Risk_Flags", flags)

    tier_label_map = {
        2: "Tier 2 (Underperformer)",
        3: "Tier 3 (Standard)",
        4: "Tier 4 (High Performer)",
    }
    results_df.insert(
        3, "Performance_Tier_Label", results_df["Predicted_PerformanceRating"].map(tier_label_map)
    )

    return results_df, blended_probs


# --- SIDEBAR: GUIDELINES & ARCHITECTURE ---
with st.sidebar:
    st.title("User Guide & Context")
    st.markdown(
        """
    ### **Business Objective**
    Mitigate service escalations and sustain INX's top employer standing through objective, data-backed talent performance analytics.

    ---
    ### **Performance Tiers**
    *  **Tier 2 (Underperformer):** Falls short of expectations; priority for empathetic diagnostic check-ins and mentorship.
    *  **Tier 3 (Standard):** Baseline contributor meeting operational benchmarks (~73% baseline).
    *  **Tier 4 (High Performer):** Top talent exceeding benchmarks; key candidate for accelerated tracks and retention equity.

    ---
    ### **Navigation**
    Use the **Single Profile** tab for detailed individual diagnostics, or switch to **Batch Roster** to score an entire department via CSV/Excel.
    """
    )

# --- MAIN PAGE HEADER ---
st.title("INX Future Inc.Employee Performance Decision Engine")
st.caption(
    "Powered by an Optimized Soft-Voting Ensemble (90% Tuned Random Forest + 10% XGBoost | Test Accuracy: 94.0%)"
)

tab_single, tab_batch = st.tabs(
    ["Single Profile Evaluation", " Batch Roster Evaluation"]
)

# TAB 1: SINGLE PROFILE EVALUATION

with tab_single:
    with st.expander("How Single Evaluation Works"):
        st.write(
            "Enter an employee's professional, compensation, and sentiment scores below to generate an immediate predicted performance rating, probability breakdown, and recommended HR actions."
        )

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
            job_level = st.slider(
                "Job Level (1 = Entry, 5 = Executive)", 1, 5, 2
            )
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
            years_company = st.number_input(
                "Tenure at Company (Years)", 0, 40, 7
            )
            years_in_role = st.number_input("Years in Current Role", 0, 30, 4)
            years_promotion = st.number_input(
                "Years Since Last Promotion", 0, 20, 2
            )
            years_manager = st.number_input(
                "Years With Current Manager", 0, 20, 3
            )

            st.caption("Sentiment Ratings (1 = Low, 4 = Very High)")
            env_satisfaction = st.slider(
                "Environment Satisfaction",
                1,
                4,
                3,
                help="Strongest model driver.",
            )
            job_involvement = st.slider("Job Involvement", 1, 4, 3)
            job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
            rel_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
            work_life = st.slider("Work-Life Balance", 1, 4, 3)

        submit = st.form_submit_button(
            " Run Performance Evaluation",
            use_container_width=True,
            type="primary",
        )

    if submit:
        travel_map = {
            "Non-Travel": 0,
            "Travel_Rarely": 1,
            "Travel_Frequently": 2,
        }
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
        input_df[numerical_columns] = scaler.transform(
            input_df[numerical_columns]
        )

        rf_probs = rf_model.predict_proba(input_df)
        xgb_probs = xgb_model.predict_proba(input_df)

        w_rf = weights.get("rf", 0.9)
        w_xgb = weights.get("xgb", 0.1)
        blended_probs = (w_rf * rf_probs) + (w_xgb * xgb_probs)
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
                    "**High-Value Talent:** Profile exhibits high output and top-tier organizational contribution."
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

        st.markdown("---")
        st.subheader("3. Strategic HR & Management Recommendations")

        if predicted_tier == 2:
            st.error(
                """
            **Priority Action Plan for Tier 2 Profile:**
            * **Root-Cause Diagnostic:** Conduct an empathetic 1-on-1 discovery session to uncover tooling blockers, process friction, or burnout.
            * **Targeted Upskilling:** Pair the employee with a senior mentor and provide focused training.
            * **Quarterly Milestones:** Establish transparent 30-60-90 day performance goals.
            """
            )
        elif predicted_tier == 3:
            st.info(
                """
            **Growth & Progression Plan for Tier 3 Profile:**
            * **Role Expansion:** Offer opportunities to take ownership of cross-departmental deliverables.
            * **Promotion Readiness:** Check whether tenure in role > 3 years indicates approaching career stagnation.
            * **Engagement Check-ins:** Maintain regular touchpoints on workplace sentiment.
            """
            )
        else:
            st.success(
                """
            **Retention & Leadership Plan for Tier 4 Profile:**
            * **Leadership Track:** Identify candidate for team mentorship and strategic initiatives.
            * **Competitive Merit Adjustment:** Ensure salary hike and retention incentives reflect top-tier market value.
            * **Autonomous Ownership:** Grant ownership over high-impact deliverables.
            """
            )

        st.markdown("##### Specific Factor Observations for this Record:")
        alerts = []
        if env_satisfaction <= 2:
            alerts.append(
                "**Low Environment Satisfaction:** Strongest statistical driver of performance decline. Investigate team dynamics or workplace friction."
            )
        if years_promotion >= 3 and years_in_role >= 4:
            alerts.append(
                "**Career Stagnation Alert:** Over 3 years without a promotion combined with long role tenure creates downward pressure on engagement."
            )
        if salary_hike < 13:
            alerts.append(
                "**Below-Average Compensation Growth:** Recent salary hikes below 13% correlate with reduced delivery velocity."
            )
        if work_life <= 2:
            alerts.append(
                "**Work-Life Balance Deficit:** Low work-life sentiment indicates elevated risk of burnout."
            )

        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.caption(
                "No critical negative risk indicators detected across compensation, environment sentiment, or promotion tenure."
            )

# TAB 2: BATCH ROSTER EVALUATION

with tab_batch:
    st.subheader("Bulk Department / Roster Evaluation")
    st.markdown(
        """
    Upload an employee roster (`.csv` or `.xlsx`) to forecast performance tiers and diagnose operational risks for entire cohorts simultaneously.
    """
    )

    uploaded_file = st.file_uploader(
        "Upload Employee Data File",
        type=["csv", "xlsx", "xls"],
        help="Upload files matching the standard INX HR schema (e.g., Age, JobRole, EnvironmentSatisfaction, etc.)",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_batch_df = pd.read_csv(uploaded_file)
            else:
                raw_batch_df = pd.read_excel(uploaded_file)

            st.success(
                f"Successfully loaded `{uploaded_file.name}` ({len(raw_batch_df)} employee records)."
            )

            with st.spinner("Processing roster through Ensemble Model..."):
                scored_df, raw_probs = process_and_predict_batch(raw_batch_df)

            # High-level Metrics
            t2_count = (scored_df["Predicted_PerformanceRating"] == 2).sum()
            t3_count = (scored_df["Predicted_PerformanceRating"] == 3).sum()
            t4_count = (scored_df["Predicted_PerformanceRating"] == 4).sum()
            total_n = len(scored_df)

            st.markdown("### Roster Performance Distribution")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Scored", f"{total_n}")
            m2.metric(
                "Tier 2 (Underperformers)",
                f"{t2_count}",
                f"{(t2_count/total_n)*100:.1f}%",
                delta_color="inverse",
            )
            m3.metric(
                "Tier 3 (Standard)",
                f"{t3_count}",
                f"{(t3_count/total_n)*100:.1f}%",
            )
            m4.metric(
                "Tier 4 (High Performers)",
                f"{t4_count}",
                f"{(t4_count/total_n)*100:.1f}%",
            )

            # Visual breakdown charts
            ch_col1, ch_col2 = st.columns([1, 1.5])
            with ch_col1:
                tier_counts = (
                    scored_df["Performance_Tier_Label"]
                    .value_counts()
                    .reset_index()
                )
                tier_counts.columns = ["Tier", "Count"]
                st.bar_chart(tier_counts.set_index("Tier"))

            with ch_col2:
                # Top Risk Indicators in Roster
                st.markdown("**Top Operational Risks Identified in Roster:**")
                all_flags = [
                    f.strip()
                    for sublist in scored_df["Operational_Risk_Flags"].str.split(
                        ";"
                    )
                    for f in sublist
                    if f.strip() != "No Critical Flags"
                ]
                if all_flags:
                    flag_summary = (
                        pd.Series(all_flags)
                        .value_counts()
                        .rename_axis("Risk Factor")
                        .reset_index(name="Affected Employees")
                    )
                    st.dataframe(flag_summary, use_container_width=True)
                else:
                    st.info("No elevated risk indicators detected in cohort.")

            # Filter & Display Table
            st.markdown("---")
            st.subheader(" Scored Roster Table")

            filter_tier = st.multiselect(
                "Filter Table by Performance Tier:",
                options=[
                    "Tier 2 (Underperformer)",
                    "Tier 3 (Standard)",
                    "Tier 4 (High Performer)",
                ],
                default=[
                    "Tier 2 (Underperformer)",
                    "Tier 3 (Standard)",
                    "Tier 4 (High Performer)",
                ],
            )

            display_df = scored_df[
                scored_df["Performance_Tier_Label"].isin(filter_tier)
            ]
            st.dataframe(display_df, use_container_width=True)

            # Download CSV Button
            csv_buffer = io.BytesIO()
            display_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label=" Download Scored Roster with Predictions & Risk Flags (CSV)",
                data=csv_buffer.getvalue(),
                file_name="INX_Scored_Employee_Roster.csv",
                mime="text/csv",
                type="primary",
            )

        except Exception as e:
            st.error(f"Error evaluating batch roster: {e}")
            st.info(
                "Ensure the uploaded file contains columns compatible with the employee dataset."
            )
    else:
        st.info(" Upload a CSV or Excel roster above to run batch evaluation.")