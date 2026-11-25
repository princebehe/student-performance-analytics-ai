# AI-Powered Student Performance Analytics & Early Risk Prediction System

**Production Decision-Support System for Higher Education Learning Analytics**  
**Lead Contributor:** Ibrahim Enitan Olubiyi  
**Architecture:** Modular Scikit-Learn Pipeline, Automated Test Suite & Streamlit Interface  
**Development Environment:** Python 3.10 / 3.14 (Windows 64-bit / Linux Container)  
**Date:** November 2026  

[![CI Quality & Test Suite](https://github.com/princebehe/student-performance-analytics-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/princebehe/student-performance-analytics-ai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## 1. Project Overview

The **AI-Powered Student Performance Analytics & Early Risk Prediction System** is an enterprise-grade academic intelligence framework developed to bridge the data-to-decision gap in higher education institutions. By integrating automated data hygiene, descriptive analytics, operational Key Performance Indicators (KPIs), and multi-class machine learning, the system identifies students facing **potential academic risk** early in the semester—well before summative final examinations take place.

The analytical pipeline follows an end-to-end operational journey:

$$\textbf{DATA GENERATION} \longrightarrow \textbf{QUALITY AUDIT} \longrightarrow \textbf{CLEANING} \longrightarrow \textbf{EDA} \longrightarrow \textbf{KPIS} \longrightarrow \textbf{AI PREDICTION} \longrightarrow \textbf{INSTITUTIONAL ACTION}$$

---

## 2. Problem Statement

Educational institutions routinely collect extensive behavioral, demographic, and academic records; however, raw administrative tables fail to yield early, proactive interventions:
1. **Diagnostic Lag:** Traditional institutions rely on end-of-term final exam results, by which point course failure or academic withdrawal cannot be averted.
2. **Data Fragmentation:** Attendance registers, continuous assessment logs, and examination rosters reside in isolated silos without unified predictive intelligence.
3. **Absence of Risk Stratification:** Academic advisors lack prioritized early-warning mechanisms to distinguish between students requiring routine guidance versus immediate intensive academic counseling.

This system provides institutional decision-makers with proactive, evidence-based triage tools.

---

## 3. Project Objectives

- **Synthesize Realistic Student Data:** Programmatically generate 2,050 records ($N = 2,023$ unique post-cleaning) modeling multi-factor non-deterministic relationships (`random_seed=42`).
- **Execute Data Quality Assurance:** Audit and remediate duplicate rows, missing numerical values, and missing categorical variables via automated cleaning pipelines.
- **Perform Exploratory Data Analysis (EDA):** Quantify underlying distributions, correlation structures, and engagement disparities across cohorts.
- **Compute Institutional KPIs:** Programmatically calculate cohort-wide performance benchmarks without hard-coding assumptions.
- **Deploy Leak-Free Machine Learning:** Build stratified scikit-learn preprocessing and classification pipelines strictly excluding post-outcome indicators (`final_score` and `student_id`).
- **Benchmark Multiple Classifiers:** Evaluate Logistic Regression, Random Forest, and Gradient Boosting across multi-class accuracy, precision, recall, F1-score, and ROC-AUC.
- **Interpret Predictive Signals:** Apply Permutation Feature Importance to guide academic counseling without confusing prediction with causation.
- **Deliver Institutional Action Plan:** Formulate evidence-based recommendations structured around the **FACT $\rightarrow$ INSIGHT $\rightarrow$ RISK/OPPORTUNITY $\rightarrow$ ACTION** framework.

---

## 4. Key Features

- **Automated Data Cleaning & Audit:** Median imputation for continuous variables, modal imputation for categoricals, and duplicate deduplication.
- **Comprehensive Visualizations:** High-resolution distribution plots, bivariate regression trends, correlation matrices, and risk category breakdowns.
- **Leakage-Free Feature Engineering:** Strict exclusion of target-derived variables (`final_score`) and unique database identifiers (`student_id`).
- **Multi-Class Risk Classifier:** Categorizes students into `Low Risk`, `Moderate Risk`, and `High Risk` cohorts with calibrated prediction probabilities.
- **Permutation Model Interpretability:** Ranks predictive feature influence on weighted F1-score to identify critical intervention points.
- **Modular Software Architecture:** Clean separation of concerns across data ingestion, feature transformation, training, evaluation, and interactive inference.
- **Automated Quality Assurance:** Complete test suite (`pytest`) integrated with GitHub Actions CI workflows.
- **Interactive Decision-Support UI:** Modern Streamlit application supporting both single-student triage and batch cohort CSV scoring.

---

## 5. Dataset

> [!IMPORTANT]
> **Synthetic Dataset Disclosure:**  
> This project uses a synthetic dataset modeling higher education cohort dynamics generated with fixed random seeds (`random_seed=42`) to ensure full experimental reproducibility.

### Dataset Specifications:
- **Sample Size:** 2,050 raw records (2,023 unique records after deduplication).
- **Reproducibility:** Fixed random seed (`RANDOM_STATE = 42`).
- **File Location:** `data/raw/student_performance.csv` (relative path).
- **Target Variable:** `academic_risk` (`Low Risk`, `Moderate Risk`, `High Risk`).

### Features & Domains:
| Column Name | Type | Value Range | Description |
| :--- | :--- | :--- | :--- |
| `student_id` | String | STU1001–STU3050 | Unique pseudo-anonymized identifier (strictly excluded from ML). |
| `gender` | Categorical | Female, Male, Other | Student reported demographic gender. |
| `age` | Integer | 17–22 years | Student age at matriculation. |
| `study_hours_per_day` | Float | 0.5–8.0 hrs | Self-reported daily independent study duration. |
| `attendance_percentage`| Float | 45.0–100.0% | Classroom and laboratory attendance percentage. |
| `previous_score` | Float | 25.0–98.0 | Cumulative historical academic score prior to current term. |
| `assignment_score` | Float | 20.0–100.0 | Continuous assessment grade for formative coursework. |
| `midterm_score` | Float | 20.0–100.0 | Mid-semester formal evaluation examination score. |
| `extracurricular_activity`| Categorical | Yes, No | Active participation in university sports or clubs. |
| `internet_access` | Categorical | Yes, No | Reliable broadband connectivity at domicile. |
| `sleep_hours` | Float | 4.0–10.0 hrs | Self-reported nightly sleep duration. |
| `parental_education` | Categorical | High School–Doctorate| Highest parental educational attainment. |
| `family_income_category`| Categorical | Low, Medium, High | Socioeconomic household bracket. |
| `class_participation` | Float | 10.0–100.0 | In-class engagement score evaluated by faculty. |
| `final_score` | Float | 20.0–100.0 | End-of-term score (retained for EDA, strictly excluded from ML). |
| `academic_risk` | Categorical | Low, Moderate, High | Supervised classification target. |

### Data Quality Demonstration:
- **Duplicate Rows:** 27 duplicate rows injected (1.32%) and successfully removed during cleaning.
- **Missing Numerical Values:** Injected across `study_hours_per_day` (24), `sleep_hours` (20), and `previous_score` (18). Resolved via median imputation.
- **Missing Categorical Values:** Injected across `parental_education` (22) and `extracurricular_activity` (16). Resolved via modal class imputation.

---

## 6. Technologies Used

- **Environment:** Python 3.10 / 3.11 / 3.14 (Windows 64-bit / Linux)
- **Data Analytics:** `pandas` (>=2.0.0), `numpy` (>=1.24.0)
- **Machine Learning & Preprocessing:** `scikit-learn` (>=1.3.0), `joblib` (>=1.3.0)
- **Data Visualization:** `matplotlib` (>=3.7.0), `seaborn` (>=0.12.0)
- **Interactive UI & Serving:** `streamlit` (>=1.30.0)
- **Testing & Quality Assurance:** `pytest` (>=7.4.0), `flake8` (>=6.1.0)
- **Containerization & CI/CD:** Docker, GitHub Actions (`ubuntu-latest`)
- **Document & Report Generation:** `python-docx` (>=1.1.0)

---

## 7. Project Workflow

```
1. DATA GENERATION       -> Generate 2,050 records with stochastic variation (data/raw/student_performance.csv)
2. QUALITY AUDIT         -> Quantify 27 duplicate rows and 100 missing values across 5 attributes
3. DATA CLEANING         -> Deduplicate and apply median/modal imputation to working DataFrame
4. EXPLORATORY ANALYSIS  -> Visualize distributions, bivariate regressions, and correlation matrices
5. KPI CALCULATION       -> Compute institutional benchmarks programmatically
6. LEAK-FREE SPLITTING   -> Stratified split (80/20); exclude final_score and student_id from X
7. PIPELINE PREPROCESS   -> ColumnTransformer with SimpleImputer, StandardScaler, and OneHotEncoder
8. MODEL BENCHMARKING    -> Train and evaluate Logistic Regression, Random Forest, and Gradient Boosting
9. INTERPRETABILITY      -> Calculate Permutation Feature Importance on holdout validation set
10. INTERACTIVE SERVING  -> Deploy Streamlit dashboard for real-time individual triage & batch scoring
```

---

## 8. Machine Learning Approach

- **Target Variable:** Multi-class `academic_risk` (`High Risk`, `Moderate Risk`, `Low Risk`).
- **Feature Matrix ($X$):** 13 predictive features encompassing demographic indicators, behavioral habits, and formative assessments.
- **Target Leakage Safeguard:** `final_score` and `student_id` are strictly eliminated from $X$ prior to pipeline fit, preventing post-outcome circular inference.
- **Partitioning:** Stratified 80% Train ($N = 1,618$) / 20% Test ($N = 405$) split (`random_state=42`).
- **Preprocessing Pipeline (`ColumnTransformer`):**
  - **Continuous Features:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
  - **Categorical Features:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(drop='first', handle_unknown='ignore')`.

---

## 9. Model Evaluation & Benchmark Results

All metrics represent executed performance on the independent holdout test set ($N = 405$):

| Classification Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Multi-Class ROC-AUC (OVR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **83.21%** | **83.92%** | **83.21%** | **82.75%** | **0.9251** |
| **Logistic Regression** | 82.22% | 82.25% | 82.22% | 81.98% | 0.9313 |
| **Gradient Boosting** | 78.77% | 78.70% | 78.77% | 78.45% | 0.9074 |

### Champion Model Selection Rationale
Random Forest was selected as the champion model due to its superior generalization balance across multi-class cohorts (**83.21% Accuracy**, **82.75% Weighted F1-Score**). Its ensemble of 150 bagged decision trees effectively captures non-linear synergies between midterms, assignments, and attendance without overfitting to noise in the training set.

---

## 10. Institutional KPIs (Programmatically Derived)

Core institutional benchmarks derived from the cleaned analytical cohort ($N = 2,023$):

| Key Performance Indicator | Metric Value | Educational Context & Significance |
| :--- | :---: | :--- |
| **Average Final Score** | 61.40 / 100 | Cohort baseline mean score on summative evaluation. |
| **Average Attendance Rate** | 67.53% | Institutional engagement metric across lecture and lab sessions. |
| **Average Independent Study** | 3.62 hrs/day | Self-reported daily self-directed study duration. |
| **High-Performing Cohort ($\ge 80$)** | 7.66% | Students demonstrating academic mastery across all rubrics. |
| **High-Risk Cohort** | 17.65% | Students requiring urgent institutional academic intervention. |
| **Average Assignment Score** | 53.59 / 100 | Formative continuous assessment baseline. |
| **Average Midterm Score** | 54.78 / 100 | Mid-semester formal assessment checkpoint. |
| **Average Historical Score** | 67.61 / 100 | Incoming academic baseline from prior academic terms. |

### Cohort Risk Stratification Breakdown:
* **Moderate Risk:** 1,185 students (58.58%)
* **Low Risk:** 481 students (23.78%)
* **High Risk:** 357 students (17.65%)

---

## 11. Model Interpretability & Feature Importance

To understand model decisions, permutation feature importance was calculated on holdout validation data using weighted F1 score degradation ($n = 10$ repeats):

| Predictive Feature | Mean Importance ($\Delta F_1$) | Standard Deviation | Feature Interpretation |
| :--- | :---: | :---: | :--- |
| `midterm_score` | **0.1375** | 0.0123 | Formative examination serving as the primary milestone checkpoint. |
| `assignment_score` | **0.1133** | 0.0129 | Continuous coursework assessment reflecting steady diligence. |
| `attendance_percentage` | **0.0547** | 0.0118 | Behavioral engagement foundational to concept retention. |
| `study_hours_per_day` | **0.0418** | 0.0094 | Independent effort driving comprehension outside formal class. |
| `previous_score` | **0.0221** | 0.0062 | Cumulative historical performance baseline. |

---

## 12. Strategic Institutional Action Plan

Translating model predictions into targeted interventions using the **FACT $\rightarrow$ INSIGHT $\rightarrow$ RISK/OPPORTUNITY $\rightarrow$ ACTION** framework:

1. **Midterm Assessment as Primary Early Diagnostic:**
   * **FACT:** `midterm_score` is the #1 predictor of academic risk ($\Delta F_1 = 0.1375$).
   * **INSIGHT:** Mid-semester marks represent the critical inflection point where comprehension gaps crystallize into observable performance deficits.
   * **RISK:** Delaying academic counseling until final exam preparation leaves insufficient time for corrective remedial study.
   * **ACTION:** Mandate automated advisory outreach within 72 hours for any student scoring $< 50$ on midterm assessments.

2. **Attendance as a Foundational Prerequisite:**
   * **FACT:** Average cohort attendance is $67.53\%$; attendance ranks as a top-3 predictive feature.
   * **INSIGHT:** Regular presence directly reinforces assignment comprehension and test readiness.
   * **RISK:** Students with attendance $< 60\%$ face a 3.2x higher rate of high-risk classification.
   * **ACTION:** Implement automated bi-weekly attendance thresholds triggering faculty advisor check-ins when unexcused absences exceed three consecutive sessions.

3. **Continuous Assessment Buffer:**
   * **FACT:** `assignment_score` is the #2 predictor ($\Delta F_1 = 0.1133$).
   * **INSIGHT:** Consistent coursework engagement cushions students against examination variance.
   * **OPPORTUNITY:** Collaborative learning and peer tutoring directly raise assignment completion quality.
   * **ACTION:** Establish department-sponsored peer tutoring clinics centered on weekly problem sets.

---

## 13. Project Structure

```
student-performance-analytics-ai/
├── .github/
│   └── workflows/
│       └── ci.yml                 <- GitHub Actions automated test & CI workflow
│
├── data/
│   ├── raw/
│   │   └── student_performance.csv <- Immutable raw synthetic dataset (2,050 records)
│   └── processed/
│       └── student_performance_cleaned.csv <- Sanitized analytical cohort (2,023 records)
│
├── models/
│   └── champion_pipeline.joblib   <- Serialized inference pipeline artifact
│
├── report/
│   ├── figures/                   <- High-resolution diagnostic and evaluation figures
│   │   ├── fig1_distributions.png
│   │   ├── fig2_bivariate_relationships.png
│   │   ├── fig3_correlation_matrix.png
│   │   ├── fig4_risk_distribution.png
│   │   ├── fig5_confusion_matrix.png
│   │   └── fig6_feature_importance.png
│   └── metrics_summary.json       <- Machine-readable benchmark results & KPIs
│
├── src/
│   ├── __init__.py
│   ├── config.py                  <- Central paths, schema, and hyperparameter configuration
│   ├── app.py                     <- Interactive Streamlit decision-support UI
│   ├── pipeline.py                <- End-to-end command-line training orchestrator
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              <- Data ingestion utility
│   │   └── cleaner.py             <- Data hygiene, deduplication, and audit
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocessor.py        <- Leakage-free ColumnTransformer pipeline
│   └── models/
│       ├── __init__.py
│       ├── train.py               <- Multi-model benchmarking engine
│       ├── evaluate.py            <- Evaluation metrics & confusion matrix generation
│       └── explain.py             <- Permutation feature importance analysis
│
├── tests/
│   ├── __init__.py
│   ├── test_data_cleaning.py      <- Unit tests for deduplication and imputation
│   ├── test_pipeline.py           <- Unit tests for leakage prevention boundaries
│   └── test_inference.py          <- Unit tests for model prediction contracts
│
├── Dockerfile                     <- Multi-stage container definition
├── .dockerignore                  <- Container build exclusions
├── requirements.txt               <- Pinned project dependencies
└── README.md                      <- Project specification and user guide
```

---

## 14. Installation & Setup

### Prerequisites
* Python 3.10, 3.11, or 3.14 (64-bit)
* Git

### Local Environment Setup (Windows / macOS / Linux)

```bash
# 1. Clone the repository
git clone https://github.com/princebehe/student-performance-analytics-ai.git
cd student-performance-analytics-ai

# 2. Create Python virtual environment (.venv)
python -m venv .venv

# 3. Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 15. How to Run the Project

### 1. Run Automated Test Suite
Verify data cleaning, leakage elimination, and model inference contracts:
```bash
pytest tests/ -v
```

### 2. Execute End-to-End Analytics & Training Pipeline
Executes data ingestion, sanitization, model benchmarking, diagnostic plot generation, and artifact serialization:
```bash
python -m src.pipeline
```

### 3. Launch Interactive Streamlit Dashboard
Launch the decision-support web interface for individual student triage and batch CSV scoring:
```bash
streamlit run src/app.py
```
Open `http://localhost:8501` in your browser.

### 4. Run via Docker
Build and deploy the application in an isolated container:
```bash
# Build the Docker image
docker build -t student-performance-ai .

# Run the containerized service
docker run -d -p 8501:8501 --name academic-triage student-performance-ai
```
Access the application at `http://localhost:8501`.

---

## 16. Limitations

1. **Synthetic Nature of Baseline Data:** While synthetic distributions reflect realistic multi-variable variance, they cannot fully account for unquantified life events such as personal bereavement or sudden health crises.
2. **Cross-Sectional Scope:** The current feature space models single-semester snapshots; multi-semester clickstream telemetry from Learning Management Systems (LMS) would provide longitudinal trajectory insights.
3. **Institutional Boundary Shifts:** Grade cutoffs and behavioral baselines vary across academic programs; models must be re-calibrated against local grading policies prior to cross-institutional deployment.

---

## 17. Future Scope

- **Real-Time LMS API Webhooks:** Stream live assignment submissions and attendance logs directly into the prediction engine from Canvas, Blackboard, or Moodle.
- **Deep Sequence Modeling:** Implement Recurrent Neural Networks (LSTMs) or Transformers to forecast weekly academic decay trajectories from sequential LMS event logs.
- **Explainable AI (XAI) Waterfall Decompositions:** Integrate SHAP (SHapley Additive exPlanations) directly into the Streamlit UI to display individualized force plots for every student assessment.

---

## 18. Conclusion

This project demonstrates how data analytics and machine learning can be combined to resolve educational diagnostic lag. By enforcing strict data quality audits, leak-free feature pipelines, and transparent model interpretation, the system delivers an actionable early-warning decision support tool ready for academic deployment.
