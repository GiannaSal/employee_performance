# src/models/export_pipeline.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def export():
    # 1. Resolve paths dynamically relative to script location
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    data_path = os.path.join(
        base_dir,
        "data",
        "raw",
        "INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls",
    )
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    output_path = os.path.join(models_dir, "inx_performance_artifacts.joblib")

    print(f"Loading data from: {data_path}")
    df = pd.read_excel(data_path)

    # 2. Drop unneeded features
    df = df.drop(
        columns=["Attrition", "OverTime", "EmpDepartment"], errors="ignore"
    )

    # 3. Categorical encoding
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
        df[col] = df[col].astype(int)

    travel_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    df["BusinessTravelFrequency"] = df["BusinessTravelFrequency"].map(
        travel_map
    )
    df["Gender"] = df["Gender"].replace({"Female": 0, "Male": 1}).astype("int8")

    nominal_cols = ["EducationBackground", "EmpJobRole", "MaritalStatus"]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)

    # 4. Feature scaling
    numerical_columns = [
        "Age",
        "DistanceFromHome",
        "EmpHourlyRate",
        "NumCompaniesWorked",
        "EmpLastSalaryHikePercent",
        "TotalWorkExperienceInYears",
        "TrainingTimesLastYear",
        "ExperienceYearsAtThisCompany",
        "ExperienceYearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ]

    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # 5. Prepare feature matrix X and target y
    X = df.drop(columns=["PerformanceRating", "EmpNumber"], errors="ignore")
    y = df["PerformanceRating"].astype(int)

    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # 6. Fit best tuned models
    print("Fitting Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X, y)

    print("Fitting XGBoost...")
    y_xgb = y - 2  # XGBoost expects 0-indexed classes [0, 1, 2]
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="mlogloss",
    )
    xgb_model.fit(X, y_xgb)

    # 7. Package and save artifacts
    artifacts = {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "scaler": scaler,
        "numerical_columns": numerical_columns,
        "feature_names": X.columns.tolist(),
        "weights": {"rf": 0.9, "xgb": 0.1},
    }

    joblib.dump(artifacts, output_path)
    print(f"Artifacts successfully saved to: {output_path}")


if __name__ == "__main__":
    export()