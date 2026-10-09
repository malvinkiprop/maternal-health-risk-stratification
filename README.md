# Explainable AI for Maternal Health Risk Stratification in Kenya

> **Research prototype for analysis and decision support.** This project explores machine-learning approaches for identifying patterns associated with missed timely postnatal care (PNC) using Kenya Demographic and Health Survey (KDHS) 2022 data. It is not a clinical diagnostic tool and must never be used to deny, delay, or restrict care.

## Overview

Postnatal care is an important opportunity to assess the health of mothers and newborns after childbirth and connect families with appropriate services. Yet not everyone receives care at the recommended time. This project uses structured survey data to study whether machine-learning models can distinguish records associated with timely PNC from records associated with missed timely PNC.

The project has two main aims:

1. **Predictive modelling:** train and evaluate classification models that estimate the likelihood of missed timely PNC.
2. **Explainability:** inspect model behaviour using SHAP (SHapley Additive exPlanations), making it easier to examine which recorded features influence predictions.

The notebook follows an end-to-end workflow inspired by CRISP-DM: understanding the problem, loading and preparing data, exploring patterns, training models, evaluating performance, interpreting predictions, and discussing responsible use. This is a predictive analysis, not a causal study: a feature associated with a prediction is not necessarily a cause of missed care.

## Research question and objectives

**Main question:** How well can machine-learning models identify patterns associated with missed timely postnatal care using the KDHS 2022 data prepared in this project?

**Objectives**
- Prepare a modelling dataset from the relevant survey files.
- Explore the outcome distribution and relationships between selected predictors and the PNC outcome.
- Train and tune classification models, including XGBoost.
- Evaluate performance using precision, recall, F1, PR-AUC, and ROC-AUC.
- Use SHAP to examine global feature importance and contributions to individual predictions.
- Document limitations and safeguards needed before any real-world application.

## Data source and study population

The analysis uses data from the **Kenya Demographic and Health Survey (KDHS) 2022**. The notebook prepares a final modelling dataset containing **10,505 women**. The reported outcome distribution is approximately **74.4% timely PNC** and **25.6% missed timely PNC**.

The input consists of structured survey variables, not medical images or free-text clinical notes. Survey definitions, coverage, missingness, recall, and data-collection methods all affect how findings should be interpreted. Consult the notebook for the precise variables and transformations used in the modelling table.

The raw DHS survey and GPS archives are not included in this repository. Obtain any required source files separately and comply with the DHS Program's access and data-use conditions.

## Target outcome

This is a **binary classification** task with two outcome classes:

- **Timely PNC** — the class representing timely postnatal care.
- **Missed timely PNC** — the class the model aims to identify.

When interpreting precision, recall, or F1, confirm the positive-class encoding in the notebook. Target construction should be understood from the code and survey definitions rather than inferred from a chart label alone.

## Methodology

### 1. Problem understanding
Define the maternal-health question, the target outcome, and the intended research use. The goal is to investigate patterns that may warrant further study, not to make clinical decisions about individuals.

### 2. Data loading and inspection
Load the relevant KDHS source files and inspect their structure, fields, and suitability for the analysis. The notebook expects the source archives described in [How to run the notebook](#how-to-run-the-notebook).

### 3. Data preparation
Prepare the modelling dataset, select predictors, construct the target, and perform the transformations required for modelling. The notebook is the authoritative source for the exact variable-level operations; review those cells before modifying or reproducing the pipeline.

### 4. Exploratory data analysis
Explore class balance and examine selected predictors against the target. Visual patterns can help formulate hypotheses, but they do not establish that a characteristic causes a woman to miss care.

### 5. Model training and tuning
Train and compare classification approaches, including a tuned **XGBoost** model. Results depend on the selected predictors, preprocessing, split strategy, model settings, and evaluation sample.

### 6. Evaluation and threshold selection
Use both threshold-independent ranking measures and threshold-dependent classification measures. The missed-timely-PNC class is the smaller class, so accuracy alone would not adequately describe model performance. The decision threshold changes the trade-off between identifying more cases and producing more false-positive flags.

### 7. Explainability
Use SHAP to examine the contribution of features to model outputs. SHAP explains the fitted model's behaviour, not the causal mechanisms behind postnatal-care access.

### 8. Interpretation
Interpret metrics and feature explanations in the context of survey limitations. Independent validation, subgroup analysis, and review by maternal-health specialists would be needed before considering operational use.

## Model evaluation metrics

| Metric | Interpretation |
|---|---|
| **Precision** | Of records predicted as missed timely PNC, the proportion that truly belong to that class. Lower precision means more false-positive flags among predicted cases. |
| **Recall** | Of records that truly belong to the missed-timely-PNC class, the proportion identified by the model. Lower recall means more missed cases. |
| **F1 score** | Harmonic mean of precision and recall; a single summary of their balance. |
| **PR-AUC** | Area under the precision–recall curve across thresholds. It is useful when the positive class is less common. |
| **ROC-AUC** | Measures how well the model ranks positive cases above negative cases across thresholds. It does not, by itself, prove probability calibration. |

### Reported results

The notebook reports the following results for the tuned XGBoost model.

**At a classification threshold of 0.50**

| Metric | Result |
|---|---:|
| PR-AUC | 0.628 |
| ROC-AUC | 0.790 |
| Precision | 0.590 |
| Recall | 0.610 |
| F1 score | 0.600 |

**At the F1-oriented threshold of 0.458**

| Metric | Result |
|---|---:|
| Precision | 0.562 |
| Recall | 0.658 |
| F1 score | 0.606 |

At threshold 0.458, recall and F1 are higher, while precision is lower. The model identifies a larger share of missed-timely-PNC records, but a greater share of its positive predictions are false positives. This is a trade-off, not an unconditional improvement. Threshold choice should reflect the intended workflow and the relative consequences of false positives and false negatives.

These are the notebook's reported evaluation results, not evidence of clinical benefit or guaranteed performance on future or geographically different populations. They should be interpreted alongside the notebook's split strategy and evaluation code.

## Explainable AI with SHAP

SHAP helps describe how a fitted model uses input features.

- **Global explanations** can summarise which features have comparatively large influence across evaluated records.
- **Direction of contribution** can indicate whether a feature value pushes a model output toward or away from the positive class.
- **Individual explanations** can break down contributions for a particular prediction, where supported by the notebook's plots.

These explanations are conditional on the model and data. They are not proof of causation, clinical reasons, or evidence that changing a feature would change an outcome. Correlated predictors and unmeasured factors can complicate interpretation.

## How to run the notebook

### Requirements

Python and the libraries used in the notebook, including Jupyter, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, and Seaborn.

### 1. Clone the repository

```bash
git clone https://github.com/malvinkiprop/maternal-health-risk-stratification.git
cd maternal-health-risk-stratification
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install the packages

```bash
pip install jupyter pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

For strict reproducibility, record and use tested dependency versions in a `requirements.txt` file.

### 4. Obtain the data separately

The notebook expects these archives under `data/raw/`:

```text
data/raw/KENR8CDT.zip
data/raw/KEGE8AFL.zip
```

They are not bundled here. Obtain the appropriate KDHS 2022 files through the official DHS Program process and follow its terms. Do not commit restricted survey data, GPS data, personal information, passwords, or access credentials to a public repository.

### 5. Run the notebook

```bash
jupyter notebook
```

Open `notebooks/maternal_health_risk_stratification.ipynb` and execute cells in order. Ensure the data files are located where the notebook expects them. If you change file locations, document the changes and check that all data-preparation steps still run correctly.

**Reproducibility note:** results can vary with input data, package versions, preprocessing, random seeds, train/test splits, and model settings. Review code and outputs before comparing results.

## Repository structure

```text
maternal-health-risk-stratification/
├── README.md
└── notebooks/
    └── maternal_health_risk_stratification.ipynb
```

This is the intended core layout. The raw `data/` archives are intentionally excluded from the public repository. If scripts, dependency files, figures, or additional documentation are added, update this section to reflect the actual files.

## Limitations and risks

1. **Survey scope:** results reflect the data, population, time period, and definitions represented by the KDHS 2022 records used.
2. **Association is not causation:** predictive relationships do not prove why missed care occurs.
3. **Class imbalance:** the minority outcome class requires metrics beyond accuracy.
4. **Threshold sensitivity:** precision, recall, and F1 change with the decision threshold.
5. **Generalisation:** performance may differ by region, population group, facility, or time period.
6. **Potential bias:** measurement error, missingness, selection effects, and historical inequalities may be reflected in the data and predictions.
7. **Calibration and utility:** ranking metrics do not establish that predicted probabilities are calibrated or that using the model improves outcomes.
8. **Unobserved context:** survey variables cannot capture every individual, household, community, and health-service factor.
9. **External validation:** independent validation and expert review are required before any operational deployment.

## Responsible use and future work

This is a **research and educational prototype**, not medical advice or a validated clinical decision system. Model predictions must not be used to deny, delay, or restrict care. Any future use should include meaningful human oversight and a clear route to appropriate services.

Potential next steps:
- Validate the pipeline and model on suitable independent data.
- Evaluate performance, calibration, and error rates across relevant demographic and geographic subgroups.
- Review false positives and false negatives with maternal-health professionals.
- Compare against transparent baseline models and assess whether ML adds practical value.
- Document variable definitions, preprocessing, dependency versions, random seeds, and model settings.
- Assess data governance, privacy, fairness, consent, and community impact.
- Consider a pilot only after rigorous validation, appropriate approvals, and a defined human-led workflow.

## Tools and libraries

- **Python** — analysis and modelling
- **pandas / NumPy** — data manipulation and numerical operations
- **scikit-learn** — modelling utilities and evaluation metrics
- **XGBoost** — gradient-boosted decision-tree model
- **SHAP** — explainability
- **Matplotlib / Seaborn** — visualisation
- **Jupyter Notebook** — executable analysis

## Data acknowledgement

This project uses Kenya Demographic and Health Survey 2022 data. Consult the official DHS Program documentation for survey details, variable definitions, data-access requirements, and citation instructions.

## Disclaimer

This repository is provided for educational and research purposes. It does not provide medical advice or a diagnosis and has not established clinical effectiveness. Any future use in maternal-health services requires independent validation, appropriate governance, privacy protections, and human oversight.
