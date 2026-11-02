# INX Future Inc. Employee Performance Analysis

## Project Overview

This project examines employee data from INX Future Inc. to understand how performance ratings vary across employees, departments, and job roles. It also develops classification models to predict employee performance ratings using available workforce data.

The business problem describes a decline in employee performance indicators alongside an **8 percentage point drop in client satisfaction**. The project aims to give leadership clearer, data-based information for investigating performance challenges and planning targeted support. It seeks to avoid broad punitive actions that could lower morale or weaken INX’s reputation as an employer.

The models identify patterns in historical data. Their predictions do not establish why an employee received a rating or prove that changing a workplace factor will improve performance. Any use of predictions should include human review and consideration of job context.

## Project Objectives

- Describe the distribution of employee performance ratings.
- Compare ratings across departments and job roles.
- Examine which employee, workplace, and career factors are associated with performance ratings.
- Build and compare models for classifying employees into ratings 2, 3, and 4.
- Identify possible interventions for further evaluation.
- Recommend ways to monitor whether interventions improve employee and business outcomes.

## Stakeholders

- **CEO and senior management:** Use the findings to inform workforce strategy and resource allocation.
- **Department heads and delivery managers:** Investigate team-level patterns, workload issues, and service delivery challenges.
- **Human Resources:** Review employee support, training, compensation, talent development, and performance management practices.

## Business Questions

1. What is the distribution of performance ratings, and how prevalent is Rating 2?
2. How do performance ratings differ across departments and job roles?
3. Which employee characteristics are associated with performance ratings?
4. How accurately can employee ratings be classified from the available data?
5. What interventions could be considered, and how can their effectiveness be measured?

## Dataset

The analysis uses the INX Future Inc. Employee Performance dataset, with **1,200 employee records** and **28 columns**. The fields cover employee demographics, job information, travel, satisfaction and engagement, compensation changes, work history, attrition, overtime, and performance ratings. The source Excel workbook also includes a data definitions sheet.

The target variable is `PerformanceRating`. It has three observed categories:

- **Rating 2:** 194 employees (16.2%)
- **Rating 3:** 874 employees (72.8%)
- **Rating 4:** 132 employees (11.0%)

Because Rating 3 is the majority class, the target is imbalanced. A model can achieve a reasonable overall accuracy while still failing to identify employees in the smaller rating groups. This is why the analysis considers per-class recall and F1-scores alongside accuracy.

The project removes the employee identifier, `EmpNumber`, from model features. It also tests removing `Attrition`, `OverTime`, and `EmpDepartment` in a cleaned Logistic Regression model.

## Data Preparation and Exploration

The analysis reviews data types, missing values, duplicate records, descriptive statistics, category frequencies, and numerical distributions. Ordinal variables—such as satisfaction, involvement, education, job level, and performance rating—are treated as ordered categories in relevant analyses. Categorical fields are encoded for modeling.

Outliers are reviewed using both the interquartile range rule and z-scores. These checks help identify unusual values, but an outlier is not automatically an error; tenure or promotion intervals may be valid for some employees.

Exploratory analysis examines overall rating frequencies and how ratings vary by department, job role, and selected employee characteristics. Statistical analyses include chi-square tests for categorical relationships and regression models to estimate associations while accounting for other included variables. Variance inflation factors are also calculated to check for multicollinearity among model predictors.

## Modeling Approach

The project compares linear and tree-based classifiers:

- **Logistic Regression:** Establishes a linear classification baseline.
- **Class-weighted Logistic Regression:** Increases the cost assigned to errors on less common ratings.
- **SMOTE and SMOTENC:** Test synthetic oversampling approaches for the minority classes.
- **Random Forest:** Models non-linear relationships and interactions through an ensemble of decision trees.
- **XGBoost:** Uses gradient-boosted trees, including a sample-weighted version to address class imbalance.
- **Soft voting ensemble:** Combines predicted class probabilities from Random Forest and XGBoost.

Models are evaluated using a train/test split and classification reports. Metrics include:

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** Among employees predicted in a rating category, the proportion who actually belong to it.
- **Recall:** Among employees in a rating category, the proportion the model correctly identifies.
- **F1-score:** A combined measure of precision and recall.
- **Macro F1:** The average F1 across rating categories, giving each category equal weight.
- **Weighted F1:** The average F1 weighted by the number of employees in each category.
- **Confusion matrix:** Shows which rating categories are being confused with one another.

## Main Findings

### Factors associated with ratings

Environment satisfaction and salary hike percentage were among the strongest predictors in the reported regression and tree-based analyses. Work-life balance was also positively associated with performance in the regression results.

Career progression measures, including years since the last promotion and years in the current role, were associated with ratings. Department and job role also contributed to differences in model predictions.


### Logistic Regression and class balancing

The initial Logistic Regression model achieved **80.0% accuracy** and a **0.770 weighted F1-score**, but performance varied substantially by rating. Rating 3 had an F1-score of **0.88**, while Rating 2 had an F1-score of **0.23** and only **17% recall**.

Removing `Attrition`, `OverTime`, and `EmpDepartment` improved the unweighted Logistic Regression results to **84.2% accuracy**, **0.842 weighted F1**, and **0.736 macro F1**. Rating 2 recall increased to **55.2%**, while Rating 4 recall reached **70.4%**.

Class weighting increased minority-class detection further. In the reported balanced Logistic Regression result, Rating 2 recall rose to **82.8%**, and Rating 4 recall reached **74.1%**. Overall accuracy fell to **74.6%**, reflecting more Rating 3 employees being classified as Rating 2 or 4. This trade-off may be useful when missing employees who could benefit from support is considered more costly than referring some borderline cases for review.

Standard SMOTE did not improve the minority-class F1 results in the reported comparison. SMOTENC achieved a Rating 2 recall of **72.4%** and the highest Rating 2 F1-score among the tested resampling approaches (**0.575**), with **78.3% accuracy**.

### Tree-based models and ensemble

Tree-based models substantially outperformed the Logistic Regression baselines in the reported results.

| Model | Accuracy | Weighted F1 | Macro F1 | Selected class result |
|---|---:|---:|---:|---|
| Cleaned, unweighted Logistic Regression | 84.2% | 0.842 | 0.736 | Rating 2 recall: 55.2% |
| Unweighted Random Forest | 92.5% | 0.923 | 0.870 | Rating 2 recall: 93.1% |
| Balanced XGBoost | 92.1% | 0.921 | 0.871 | Rating 2 recall: 89.7%; Rating 4 recall: 77.8% |
| Soft voting ensemble | 92.9% | 0.929 | 0.881 | Rating 2 recall: 89.7%; Rating 4 F1: 0.851 |
| Tuned soft voting ensemble | 94.0% | 0.940 | 0.890 | Rating 2 recall: 93%; Rating 4 recall: 74% |

Unweighted Random Forest achieved the highest reported Rating 2 recall among the individual tree models, correctly identifying 27 of 29 Rating 2 employees. Balanced XGBoost had a stronger macro F1-score than the unweighted Random Forest in the reported model comparison and identified a larger share of Rating 4 employees.

The soft voting ensemble combined Random Forest and XGBoost probabilities. Its reported untuned result achieved **92.9% accuracy** and **0.881 macro F1**. A later tuning section reports an optimized ensemble result of **94.0% accuracy** and **0.890 macro F1**, using a 0.9 Random Forest and 0.1 XGBoost probability blend.

These numbers come from different stages of the analysis and should not be treated as directly interchangeable. The notebook’s tuning workflow selects blending weights using test-set macro F1. To obtain an unbiased final performance estimate, hyperparameters and blending weights should be selected using training or validation data, followed by one evaluation on a held-out test set.

## Recommendations for Further Investigation

- **Work environment and engagement:** Use pulse surveys and department feedback to identify workplace barriers, resource gaps, or management issues. Review concerns with employees and managers before selecting actions.
- **Compensation and recognition:** Examine whether salary increases and recognition practices are transparent, consistent, and aligned with role expectations and documented outcomes.
- **Work-life balance:** Review workloads, staffing, and flexibility in teams where employees report lower work-life balance.
- **Career progression:** Clarify promotion criteria, skills requirements, internal mobility options, and development pathways.
- **Department and role context:** Compare rating patterns alongside job responsibilities and evaluation practices before interpreting departmental or role differences as performance gaps.

These are areas for investigation, not proven remedies. Interventions should be designed with employees and evaluated for unintended effects on morale, retention, fairness, and service quality.

## Evaluating Interventions

- **Track outcomes over time:** Monitor performance rating changes, employee satisfaction, retention, and client satisfaction before and after an intervention.
- **Use pilot and comparison groups:** Where practical, introduce an intervention in selected teams and compare changes with similar teams that have not yet received it.
- **Define success measures in advance:** Specify the expected change, the evaluation period, and the data needed to measure it.
- **Review model performance periodically:** Reassess accuracy, class-specific recall, and potential differences in error rates across employee groups as new data becomes available.
- **Combine quantitative and qualitative evidence:** Discuss results with employees and managers to understand what may be driving observed changes.

A shift in model predictions alone does not establish that an intervention worked. Evaluation should use observed outcomes and an appropriate comparison.

## Tools and Reproducibility

Python and the following libraries:

- pandas, NumPy, SciPy
- Matplotlib and Seaborn
- statsmodels
- scikit-learn
- imbalanced-learn
- XGBoost


## Limitations

- The dataset contains 1,200 records and is specific to one organization.
- Performance ratings are imbalanced, and the smaller rating groups have fewer examples.
- Associations and feature importance do not establish causation.
- Ratings may reflect role-specific responsibilities or evaluation practices not represented in the data.
- Reported model results vary by modeling stage; final performance should be confirmed with a leakage-safe validation process and independent data where possible.
- Model predictions should be reviewed for fairness and used to support—not replace—professional judgment.

## Suggested Next Steps

1. Confirm the final modeling pipeline and evaluate it once on a held-out test set after tuning.
2. Validate performance on newer data before relying on the model in operational settings.
3. Review error rates across relevant employee groups and job roles.
4. Agree on safeguards and human review procedures before using individual-level predictions.
5. Pilot workplace interventions and measure employee, operational, and client outcomes.
