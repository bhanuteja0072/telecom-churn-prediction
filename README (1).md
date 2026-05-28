# Data

This project uses the **Telecom Churn Case Study** dataset from Kaggle.

- **Source**: https://www.kaggle.com/competitions/telecom-churn-case-study-hackathon-c55/data
- **Files needed**: `train.csv`, `test.csv`

## Setup

1. Download `train.csv` and `test.csv` from the Kaggle link above
2. Place both files in this `data/` folder
3. Do **not** commit these files — they are listed in `.gitignore`

## Dataset Description

The dataset contains 4 months of usage data (June–September) for high-value telecom customers in India. The task is to predict churn in the 9th month (September) using behaviour from June–August.

### Key column groups

| Prefix | Description |
|---|---|
| `arpu_*` | Average Revenue Per User per month |
| `onnet_mou_*` / `offnet_mou_*` | On-net / off-net minutes of usage |
| `roam_*` | Roaming usage |
| `loc_*` / `std_*` / `isd_*` / `spl_*` | Local / STD / ISD / special call minutes |
| `total_rech_*` | Recharge count and amount |
| `vol_*` | 2G/3G data volume (MB) |
| `monthly_*` / `sachet_*` | Monthly and sachet scheme counts |
| `churn_probability` | **Target** — 1 = churned, 0 = retained |

Suffixes `_6`, `_7`, `_8` represent June, July, August respectively.
