# 📱 Telecom Churn Prediction

Predicting high-value customer churn for an Indian telecom operator using ensemble machine learning — XGBoost and Random Forest with custom feature engineering on 4 months of usage data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Hackathon-blue)](https://www.kaggle.com/competitions/telecom-churn-case-study-hackathon-c55)

---

## 📌 Project Overview

This project tackles the telecom churn prediction problem for a high-value customer segment. The business context: in the Indian telecom market, the top 20% of customers contribute ~80% of revenue. Identifying which of these customers are likely to churn — before they do — allows targeted retention efforts.

The dataset contains monthly usage data across June–August, and the goal is to predict churn in September (month 9).

---

## 🔬 Methodology

### 1. Feature Engineering
Rather than using raw monthly columns directly, the notebook derives behavioural trend features that capture *how* usage is changing:

- **Phase drift**: `month_8 − avg(month_6, month_7)` — measures deviation from the good-phase baseline
- **Month-over-month diffs**: `month_7 − month_6` and `month_8 − month_7`
- **Monthly averages**: per-user averages across June–August

Applied across all feature groups: ARPU, on-net/off-net MOU, roaming, local/STD/ISD/special calls, recharge amounts, data volume, monthly and sachet schemes.

### 2. High-Value Customer Derivation
A `high_value_customer` flag is derived from ARPU and recharge amount thresholds, used to filter down to the most business-critical segment.

### 3. Class Imbalance Handling
Churn is rare (~8–9% positive rate). **SMOTE** (Synthetic Minority Oversampling) is applied on the training set to balance classes before model training.

### 4. Dimensionality Reduction
**PCA** is applied after scaling to reduce the high-dimensional engineered feature space while preserving variance.

### 5. Models

Two models are trained for different business objectives:

| Model | Objective | Threshold |
|---|---|---|
| **XGBoost** | Maximise accuracy (Kaggle leaderboard) | 0.70 |
| **Random Forest** | Maximise recall (business objective) | 0.36 |

The Random Forest model at threshold 0.36 is the **recommended model** — in churn prediction, missing a churner (false negative) is far more costly than a false alarm.

### 6. Ensemble Evaluation
A custom `Model_Ensemble` function handles training, cross-validation, and threshold analysis with full classification metrics: accuracy, precision, sensitivity (recall), specificity, and AUC-ROC.

---

## 📊 Key Results

| Metric | XGBoost (th=0.70) | Random Forest (th=0.36) |
|---|---|---|
| **Accuracy** | Higher | Moderate |
| **Recall** | Lower | **Highest** |
| **AUC-ROC** | Competitive | Competitive |

The Random Forest model at threshold 0.36 gives the best balance across all metrics and is used for the final business submission.

### Top Predictive Features (by importance)

1. `loc_mou_monthly` — local call minutes trend
2. `loc_mou_diff_8_7` — change in local calls in the action phase
3. `loc_mou_phase_drift` — deviation from good-phase baseline
4. `amt_per_recharge_phase_drift` — recharge amount drift
5. `total_mou_phase_drift` — total usage drift
6. `arpu_phase_drift` — revenue drift in action month

---

## 💡 Business Recommendations

- A sudden drop in MOU, ARPU, or recharge amount in the action month (August) is the strongest churn signal
- Customers who drastically reduce local call minutes are at highest risk
- Proactively target the `high_value_customer` segment with personalised retention offers before churn occurs
- Send satisfaction surveys to customers showing phase drift signals; addressing dissatisfaction early is more effective than win-back campaigns

---

## 🛠️ Technologies

| Tool | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation & feature engineering |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | Preprocessing, PCA, RFE, Random Forest, metrics |
| `xgboost` | Gradient boosting classifier |
| `imbalanced-learn` | SMOTE for class imbalance |
| Jupyter Notebook | Analysis environment |

---

## 📂 Repository Structure

```
telecom-churn-prediction/
├── notebooks/
│   └── telecom_churn_case_study.ipynb   ← full analysis
├── data/
│   └── README.md                         ← data setup instructions
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/bhanuteja0072/telecom-churn-prediction.git
cd telecom-churn-prediction
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Download the data:**

Get `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/competitions/telecom-churn-case-study-hackathon-c55/data) and place them in the `data/` folder.

**4. Run the notebook:**
```bash
jupyter notebook notebooks/telecom_churn_case_study.ipynb
```

---

## 📖 Data Source

Kaggle — [Telecom Churn Case Study Hackathon](https://www.kaggle.com/competitions/telecom-churn-case-study-hackathon-c55/data)
