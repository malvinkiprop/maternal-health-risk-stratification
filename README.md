# Explainable AI for Maternal Health Risk Stratification in Kenya

An explainable machine-learning project exploring whether Kenya Demographic and Health Survey (KDHS 2022) data can help identify women who may miss a timely postnatal care (PNC) check within the first two days after childbirth.

> **Responsible-use note:** This is an experimental research and decision-support project, not a clinical diagnostic tool. Model predictions must not be used to deny care or make consequential decisions about an individual without appropriate validation, safeguards, and human oversight.

## Project objective

Develop and evaluate an explainable binary classifier for missed timely PNC, compare a Logistic Regression baseline with XGBoost models, and use SHAP explanations to make model behaviour easier to inspect.

## Dataset and target

- **Source:** Kenya Demographic and Health Survey 2022 pregnancy/postnatal-care recode.
- **Final modelling population:** 10,505 unique women after eligibility filtering, removal of unknown-target records, and deduplication, as documented in the notebook.
- **Target:** whether a woman missed a postnatal health check within two days of childbirth.
- **Class balance:** approximately 74.4% timely PNC and 25.6% missed timely PNC in the final modelling population.
- **Geospatial data:** the GPS archive is checked for mapping-related work, but GPS coordinates are explicitly excluded from model predictors.

The raw DHS survey and GPS archives are not included in this repository. Access and use must follow DHS permissions and confidentiality conditions.

## Workflow

The notebook follows the CRISP-DM workflow and covers:

1. Business understanding and problem framing.
2. Survey data loading, eligibility filtering, target construction, validation, and deduplication.
3. Exploratory data analysis and examination of missingness and class balance.
4. Train/test splitting and preprocessing of numeric and categorical predictors.
5. Class-imbalance handling using class weights rather than synthetic respondents.
6. Comparison of Logistic Regression, untuned XGBoost, and tuned XGBoost.
7. Evaluation using PR-AUC, ROC-AUC, precision, recall, F1 score, and a confusion matrix.
8. Classification-threshold review.
9. Global and individual-level SHAP explanations.

Hyperparameter tuning uses stratified cross-validation on the training set, with average precision (PR-AUC) as the selection metric; the held-out test set is reserved for final evaluation.

## Reported evaluation results

The project summary reports the tuned XGBoost model at the default 0.5 classification threshold as follows:

| Metric | Reported result |
|---|---:|
| PR-AUC | 0.628 |
| ROC-AUC | 0.790 |
| Precision | 0.590 |
| Recall | 0.610 |
| F1 score | 0.600 |

An F1-oriented threshold of **0.458** was also reported, with precision **0.562**, recall **0.658**, and F1 **0.606**. Changing the threshold changes the trade-off between flagging more possible missed-PNC cases and creating additional false alerts. The operational threshold should be determined with relevant health-program stakeholders, not treated as a clinical cutoff.

These figures are experimental model results, not evidence of clinical effectiveness. Refer to the notebook for the evaluation workflow and detailed outputs.

## Explainability

The project uses **SHAP (SHapley Additive exPlanations)** for:

- Global summaries of which transformed features contribute to model outputs.
- Local explanations showing how features contribute to individual predicted risk scores.

SHAP can help inspect model behaviour, but it does not by itself establish causality, fairness, or clinical suitability.

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, and Seaborn.

## Run the notebook

### 1. Clone the repository

```bash
git clone https://github.com/malvinkiprop/maternal-health-risk-stratification.git
cd maternal-health-risk-stratification
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn
```

Use a Python environment with compatible versions of these packages; the notebook uses `OneHotEncoder(sparse_output=False)`, which requires a recent scikit-learn release.

### 3. Provide the required source archives

Create `data/raw/` and place the authorized DHS archives there with the exact filenames expected by the notebook:

```text
data/raw/KENR8CDT.zip
 data/raw/KEGE8AFL.zip
```

- `KENR8CDT.zip` — pregnancy and postnatal-care recode archive.
- `KEGE8AFL.zip` — GPS archive, used only for the notebook's mapping-related verification and not as model predictors.

These archives are intentionally not committed to GitHub. Obtain them through the appropriate DHS access process and comply with their terms. The notebook expects to be run from the repository directory (or a subdirectory from which it can locate `data/raw/`).

### 4. Open and run

Open `notebooks/maternal_health_risk_stratification.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab after placing the authorized data archives in the expected folder. Run the cells in order. Model training and SHAP explanations may take time and computational resources.

## Limitations and responsible use

- Results are based on survey data and may not generalize to other populations, regions, or time periods without further evaluation.
- Held-out performance does not establish clinical utility or improved maternal-health outcomes.
- Predictions should support further assessment and follow-up planning only; they must not replace professional judgement or be used to withhold services.
- Survey data must be stored, processed, and shared according to the applicable permissions and confidentiality requirements.
- A model score is not a causal explanation or a diagnosis.

## Repository contents

- `README.md` — project overview, methodology, reported results, setup, and limitations.
- `notebooks/maternal_health_risk_stratification.ipynb` — end-to-end CRISP-DM notebook.
