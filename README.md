# TrendPulse: YouTube Trending Video Analytics & Popularity Prediction

> **Course Project:** Exploratory Data Analysis & Predictive Machine Learning  
> **Status:** Review-2 Complete (Completion Estimate: **~85%**)

---

## 📌 Project Overview

**TrendPulse** is an end-to-end data analytics and machine learning project designed to analyze YouTube video engagement dynamics and predict video popularity levels. By consolidating multi-source data from the **YouTube Data API v3** and historical **Kaggle Trending Datasets**, TrendPulse investigates the core drivers of audience engagement, content category dominance, upload timing strategies, and viral outlier behavior.

### 🏗️ Proposed System Architecture

```text
YouTube API v3 + Kaggle Dataset
               │
               ▼
    [Phase 1 & 2: Audit & Quality Assessment]
               │
               ▼
    [Phase 3 & 4: Cleaning & Master Consolidation]
               │
               ▼
    [Phase 5: Feature Engineering]
               │
               ▼
    [Phase 6: Exploratory Data Analysis (EDA)]
               │
               ▼
    [Phase 7: Machine Learning & Popularity Prediction]
               │
               ▼
    [Insights, Charts & Model Evaluation]
```

---

## 📂 Project Directory Structure

```text
EDA Project/
│
├── data/
│   ├── raw/                        # Original raw Kaggle CSVs and API exports
│   │   ├── INvideos_cleaned.csv
│   │   ├── youtube_api_by_category.csv
│   │   └── youtube_api_trending_raw.csv
│   ├── cleaned/                    # Cleaned & standardized CSV datasets
│   │   ├── kaggle_cleaned.csv
│   │   ├── api_cleaned.csv
│   │   └── youtube_master.csv      # Consolidated master dataset (17,357 records)
│   └── processed/                  # Feature-engineered dataset
│       └── youtube_features.csv    # Master dataset with 27 calculated features
│
├── src/
│   ├── api/                        # YouTube API v3 integration client & category maps
│   ├── parsing/                    # Raw JSON response parser
│   ├── preprocessing/              # Phase 1-4: Data audit, cleaning & consolidation pipeline
│   │   └── clean_data.py
│   ├── feature_engineering/        # Phase 5: Feature generation pipeline
│   │   └── build_features.py
│   ├── eda/                        # Phase 6: Automated 7-part EDA visual pipeline
│   │   └── run_eda.py
│   └── models/                     # Phase 7: ML training, evaluation & metrics pipeline
│       └── train_models.py
│
├── reports/
│   └── data_quality_report.json    # Phase 1 & 2 dataset audit results
│
├── results/
│   ├── eda/                        # High-resolution generated EDA plots (PNGs)
│   │   ├── 01_view_distribution.png
│   │   ├── 02_category_distribution.png
│   │   ├── 03_engagement_by_category.png
│   │   ├── 04_upload_day_analysis.png
│   │   ├── 05_upload_hour_analysis.png
│   │   ├── 06_correlation_heatmap.png
│   │   ├── 07_outlier_detection.png
│   │   └── eda_summary.json
│   └── models/                     # Model confusion matrices & feature importance
│       ├── confusion_matrices.png
│       ├── feature_importance_rf.png
│       ├── model_metrics.json
│       └── sample_predictions.csv
│
├── trendpulse.md                   # Master project requirement specification & roadmap
└── README.md                       # Main project documentation
```

---

## 📊 Phase 1-4: Data Audit, Cleaning & Consolidation

### 🔍 Data Quality Audit Summary

| Dataset Source | Raw Records | Columns | Missing Values Handled | Duplicate Video IDs Removed | Cleaned Records |
|---|---|---|---|---|---|
| **Kaggle (IN Videos)** | 32,272 | 20 | Description (524 nulls) | 15,966 duplicates | 16,306 |
| **YouTube API (Category)** | 929 | 14 | Description (329), Likes (18) | 0 duplicates | 929 |
| **YouTube API (Trending)** | 199 | 13 | Description (5), Likes (5) | 77 duplicates (cross-feed) | 122 |
| **Consolidated Master** | **33,400** | **10 (Std)** | **0 nulls remaining** | **Deduplicated on `video_id`** | **17,357** |

### 🛠️ Preprocessing Pipeline Actions (`src/preprocessing/clean_data.py`)
1. **Schema Standardization:** Standardized raw columns into standard format: `video_id`, `title`, `description`, `channel_title`, `category_id`, `category_name`, `published_at`, `views`, `likes`, `comment_count`, `source`.
2. **Category ID Resolution:** Mapped category IDs to official YouTube Category titles using standard API mappings.
3. **Data Quality Corrections:** Handled non-ASCII encoding errors, coerced invalid numerical values to zero, converted timestamps to standard ISO datetime.
4. **Deduplication Strategy:** Sorted by view count descending before dropping duplicate `video_id` records to preserve peak engagement metrics.

---

## ⚡ Phase 5: Feature Engineering

The feature engineering pipeline (`src/feature_engineering/build_features.py`) generated 27 analytical variables saved in `data/processed/youtube_features.csv`:

### Engineered Feature Categories
- **Content Features:** `title_length` (character count), `title_word_count`, `description_length`.
- **Temporal Features:** `upload_hour` (0-23 UTC), `upload_day` (Mon-Sun), `upload_day_num` (0-6), `upload_month` (1-12), `upload_year`, `is_weekend` (0/1).
- **Recency & Age:** `video_age_days` (difference between max date and publish date).
- **Engagement Ratios & Metrics:**
  $$\text{like\_ratio} = \frac{\text{likes}}{\text{views}}$$
  $$\text{comment\_ratio} = \frac{\text{comments}}{\text{views}}$$
  $$\text{engagement\_score} = \frac{\text{likes} + \text{comment\_count}}{\text{views}} \times 100$$
- **Log Transformations:** `log_views`, `log_likes`, `log_comment_count` for handling positive skewness.

---

## 📈 Phase 6: Exploratory Data Analysis (EDA) Insights

All 7 core visual analyses specified in `trendpulse.md` were executed (`src/eda/run_eda.py`) and saved to `results/eda/`:

### 1. View Count Distribution (`01_view_distribution.png`)
- **Median Views:** 216,776 views | **Mean Views:** 747,624 views.
- **Finding:** View count exhibits extreme positive skewness; over 80% of videos are concentrated in lower view ranges (< 1M views), requiring log transformation ($\log(1 + \text{views})$) for predictive modeling.

### 2. Category Dominance (`02_category_distribution.png`)
- **Top Categories:** **Entertainment** (7,624 videos, 43.9%), followed by **News & Politics** (4,619) and **Music** (3,137).

### 3. Engagement Rate by Category (`03_engagement_by_category.png`)
- **Top Engagement Category:** **Science & Technology** yields the highest average engagement rate (**9.67%**), followed by **Education** (**8.12%**) and **Comedy** (**7.45%**), demonstrating that niche educational content drives higher interaction rates per view than mass entertainment.

### 4. Upload Day Strategy (`04_upload_day_analysis.png`)
- **Peak Upload Volume Day:** **Friday** (3,120 uploads).
- **Peak Audience Engagement Day:** **Sunday** (Highest average engagement score), showing an opportunity for creators to publish on weekends when viewer interaction propensity is highest.

### 5. Upload Hour Performance (`05_upload_hour_analysis.png`)
- **Peak Upload Volume:** **14:00 UTC** (2:00 PM UTC).
- **Peak Average Views:** **13:00 UTC**, indicating optimal publishing alignment just before peak afternoon viewing hours.

### 6. Correlation Analysis (`06_correlation_heatmap.png`)
- **Views & Likes Correlation:** $+0.85$ (Strong positive correlation).
- **Views & Comments Correlation:** $+0.62$ (Moderate positive correlation).
- **Title Length & Views:** $-0.06$ (Near-zero linear correlation), disproving the assumption that longer video titles correlate with higher view counts.

### 7. Viral Outlier Detection (`07_outlier_detection.png`)
- **99th Percentile Threshold:** **9,760,574 views**.
- **Viral Video Count:** **174 viral outliers** identified (Top 1%). Highly viral videos maintain strong absolute engagement, though their percentage engagement rate stabilizes as passive audience reach broadens.

---

## 🤖 Phase 7: Machine Learning & Popularity Prediction (Regression on log_views)

### Task Definition
Continuous view popularity prediction with target variable **`log_views`** ($\log(1 + \text{views})$) as specified in `modeltraining.md`.
- **Reason:** Reduced skewness (0.44), near-normal distribution, and stable gradient learning.
- **Features Used:**
  - *Numerical:* `title_length`, `title_word_count`, `description_length`, `upload_hour`, `upload_day_num`, `upload_month`, `upload_year`, `is_weekend`, `video_age_days`
  - *Categorical:* `category_name`, `source`
- **Features Excluded:** `video_id`, `title`, `description`, `channel_title`, `published_at` (reserved for future NLP iterations).

### Model Performance Comparison

| Model | Role | RMSE (Primary Metric) | MAE | R² Score |
|---|---|---|---|---|
| **RandomForestRegressor** | Baseline Model | **1.1050** | **0.8283** | **0.3557** |
| **XGBoostRegressor** | Main Model | **1.1104** | **0.8388** | **0.3494** |
| **CatBoostRegressor** | Final Model | **1.1214** | **0.8509** | **0.3364** |

### Generated Visualizations & Artifacts
1. **Actual vs Predicted Scatter Plots:** [`results/models/regression_actual_vs_predicted.png`](file:///c:/Users/prane/Desktop/VIT/Projects/EDA Project/results/models/regression_actual_vs_predicted.png)
2. **Residuals Distribution:** [`results/models/residuals_distribution.png`](file:///c:/Users/prane/Desktop/VIT/Projects/EDA Project/results/models/residuals_distribution.png)
3. **Feature Importance Comparison:** [`results/models/feature_importance_comparison.png`](file:///c:/Users/prane/Desktop/VIT/Projects/EDA Project/results/models/feature_importance_comparison.png)
4. **Evaluation Metrics JSON:** [`results/models/regression_metrics.json`](file:///c:/Users/prane/Desktop/VIT/Projects/EDA Project/results/models/regression_metrics.json)
5. **Sample Predictions:** [`results/models/regression_sample_predictions.csv`](file:///c:/Users/prane/Desktop/VIT/Projects/EDA Project/results/models/regression_sample_predictions.csv)

---

## 📋 Review-2 Demonstration Checklist Verification

| Requirement | Target | Status | Artifact / Output File |
|---|---|---|---|
| **Data Cleaning** | Before vs After audit | ✅ Complete | `reports/data_quality_report.json` |
| **Data Consolidation** | Single Master CSV | ✅ Complete | `data/cleaned/youtube_master.csv` |
| **Feature Engineering** | Derived metrics created | ✅ Complete | `data/processed/youtube_features.csv` |
| **EDA Visualizations** | 7 core analyses plots | ✅ Complete | `results/eda/01_*.png` to `07_*.png` |
| **Machine Learning** | RF, XGBoost & CatBoost | ✅ Complete | `results/models/regression_actual_vs_predicted.png` |
| **Sample Predictions** | Prediction output CSV | ✅ Complete | `results/models/regression_sample_predictions.csv` |
| **Project Progress** | $\ge 75\%$ complete | ✅ **~85%** | Exceeds Review-2 Requirements |

---

## 🚀 How to Run the Pipeline

Execute the end-to-end analytical pipeline sequentially using the following commands:

```bash
# 1. Clean & Consolidate Datasets (Phases 1-4)
python src/preprocessing/clean_data.py

# 2. Build Engineered Features (Phase 5)
python src/feature_engineering/build_features.py

# 3. Generate Exploratory Data Analysis & Plots (Phase 6)
python src/eda/run_eda.py

# 4. Train & Evaluate Machine Learning Regressors (Phase 7)
python src/models/train_models.py
```

---

## 🔮 Future Scope (Post Review-2)

1. **NLP Sentiment & Topic Modeling:** Extract text embeddings and sentiment polarity from video titles and descriptions.
2. **Interactive Streamlit Dashboard:** Build a real-time web dashboard allowing creators to enter video metadata and receive instant view predictions.
3. **Hyperparameter Optimization:** Conduct Optuna / GridSearchCV sweeps for XGBoost and CatBoost.
