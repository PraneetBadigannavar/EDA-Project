# TrendPulse
### YouTube Trending Video Analytics and Popularity Prediction Using Data Analytics and Machine Learning

---

# Project Overview

TrendPulse is an Exploratory Data Analysis (EDA) and Machine Learning project focused on understanding the factors that influence YouTube video performance. The system combines data collected from the YouTube Data API and publicly available Kaggle datasets to analyze video engagement, identify performance patterns, and develop predictive models capable of estimating future engagement levels.

The project is being developed primarily as an EDA course project, with machine learning serving as an extension of the analytical pipeline.

---

# Current Status

## Completed

### Literature Survey
Research papers collected on:

- YouTube popularity prediction
- Video engagement analysis
- Social media analytics
- Content recommendation systems
- Trend forecasting

### Data Collection

#### Source 1: YouTube Data API

Collected metadata including:

- Video Title
- Description
- Publish Date
- Channel Name
- View Count
- Like Count
- Comment Count
- Category Information

#### Source 2: Kaggle Dataset

Historical YouTube video datasets collected for large-scale analysis and model training.

### Proposed Architecture

```text
YouTube API + Kaggle Dataset
                │
                ▼
         Data Collection
                │
                ▼
         Data Cleaning
                │
                ▼
      Feature Engineering
                │
                ▼
 Exploratory Data Analysis
                │
                ▼
        Machine Learning
                │
                ▼
    Insights & Prediction
```

Current Completion Estimate:
~20%

---

# Review-2 Goal

Target Completion:
75%+

Faculty evaluation heavily emphasizes implementation.

The following components must be completed before Review-2:

- Data Cleaning
- Data Consolidation
- Feature Engineering
- EDA
- One Machine Learning Model

FastAPI, Dashboard, NLP, and Deployment are not priorities for Review-2.

---

# Project Directory Structure

```text
TrendPulse/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── eda/
│   └── models/
│
├── reports/
│
├── presentation/
│
├── results/
│
└── README.md
```

---

# Implementation Roadmap

## Phase 1: Dataset Understanding

Objective:
Understand structure and quality of all collected data.

Tasks:

- Load Kaggle dataset
- Load API dataset
- Identify common columns
- Create schema comparison

Output:

```text
Column Comparison Table

| Column | Kaggle | API |
|---------|---------|---------|
| title | ✓ | ✓ |
| views | ✓ | ✓ |
| likes | ✓ | ✓ |
```

Deliverable:
Data audit report.

Status:
Pending

---

## Phase 2: Data Quality Assessment

Objective:
Identify issues before preprocessing.

Tasks:

### Missing Value Analysis

Check:

- Null values
- Empty fields

### Duplicate Analysis

Check:

- Duplicate video IDs
- Duplicate records

### Data Type Validation

Verify:

- Numerical fields
- Datetime fields
- Categorical fields

Output:

```text
Rows:
Missing Values:
Duplicates:
Incorrect Datatypes:
```

Deliverable:
Data Quality Report

Status:
Pending

---

## Phase 3: Data Cleaning

Objective:
Produce a reliable dataset.

Tasks:

- Remove duplicates
- Handle missing values
- Standardize categories
- Convert dates
- Validate numerical columns
- Remove corrupted records

Output:

```text
youtube_cleaned.csv
```

Location:

```text
data/cleaned/
```

Status:
Pending

---

## Phase 4: Dataset Consolidation

Objective:
Merge Kaggle and API data.

Recommended Approach:

Keep common columns only.

Advantages:

- Faster implementation
- Cleaner dataset
- Easier EDA

Output:

```text
youtube_master.csv
```

Status:
Pending

---

## Phase 5: Feature Engineering

Objective:
Create meaningful analytical features.

Features to Create:

### Content Features

- title_length
- word_count

### Time Features

- upload_hour
- upload_day
- upload_month

### Age Features

- video_age

### Engagement Features

- like_ratio
- comment_ratio

### Derived Metrics

```text
engagement_score =
(likes + comments) / views
```

Output:

```text
youtube_features.csv
```

Location:

```text
data/processed/
```

Status:
Pending

---

## Phase 6: Exploratory Data Analysis

Objective:
Discover patterns within the dataset.

### Analysis 1

View Distribution

Question:

Are most videos concentrated in lower view ranges?

---

### Analysis 2

Category Distribution

Question:

Which categories dominate YouTube?

---

### Analysis 3

Engagement by Category

Question:

Which categories generate higher audience interaction?

---

### Analysis 4

Upload Day Analysis

Question:

Which weekday produces the highest engagement?

---

### Analysis 5

Upload Hour Analysis

Question:

What upload time performs best?

---

### Analysis 6

Correlation Analysis

Variables:

- views
- likes
- comments
- engagement_score

Output:

Correlation Heatmap

---

### Analysis 7

Outlier Detection

Identify:

- Viral videos
- Exceptional performers
- Unusual engagement behavior

Output Folder:

```text
results/eda/
```

Expected Deliverables:

- Histograms
- Bar Charts
- Scatter Plots
- Heatmaps
- Summary Observations

Status:
Pending

---

## Phase 7: Machine Learning

Objective:
Predict engagement level.

### Problem Definition

Classification Task

Output Classes:

- High Engagement
- Medium Engagement
- Low Engagement

Label Creation:

```text
Top 33%      → High
Middle 33%   → Medium
Bottom 33%   → Low
```

### Features

Potential Inputs:

- title_length
- upload_hour
- upload_day
- category
- views
- subscriber_count
- like_ratio
- comment_ratio

### Models

Model 1:

- Logistic Regression

Model 2:

- Random Forest

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Output Folder:

```text
results/models/
```

Status:
Pending

---

# Review-2 Demonstration Checklist

Faculty should be shown:

## Dataset

- Number of records
- Number of features

## Data Cleaning

- Before vs After comparison

## Feature Engineering

- Newly generated features

## EDA

At least:

- Category Analysis
- Upload Time Analysis
- Engagement Analysis
- Correlation Heatmap

## Machine Learning

At least one trained model showing:

- Accuracy
- Confusion Matrix
- Sample Predictions

---

# Future Scope (After Review-2)

Not required immediately.

### NLP

- Sentiment Analysis
- Keyword Extraction
- Topic Modeling
- TF-IDF

### Backend

- FastAPI

### Dashboard

- Streamlit

### Advanced Models

- XGBoost
- LightGBM
- Deep Learning Models

### Deployment

- Cloud Hosting
- Real-time Analytics

---

# Research Gap

Existing studies typically focus on only one aspect:

- Popularity Prediction
- Engagement Analysis
- Dashboard Visualization
- NLP Analysis

TrendPulse aims to unify:

- Data Collection
- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Machine Learning
- Visualization

within a single analytical pipeline.

---

# Immediate Next Task

Start with:

1. Load Kaggle dataset
2. Load YouTube API dataset
3. Create column comparison table
4. Generate data quality report
5. Begin data cleaning

Do not start machine learning until EDA is completed.