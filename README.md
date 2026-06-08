# Student Performance Prediction using Machine Learning

An end-to-end Machine Learning pipeline built using Python and Scikit-Learn to predict whether a student will pass (score $\ge$ 60%) or fail an exam based on behavioral and academic metrics.

## 📌 Project Overview
This repository serves as a practical implementation of supervised classification algorithms. The project utilizes a synthetic but highly realistic dataset containing 1,000 student profiles to map out variables like study hours and class attendance against academic success.

### Problem Statement
Educational institutions often struggle to identify at-risk students early enough in an academic term to offer impactful interventions. By modeling historical performance features, this project creates an automated classification tool capable of identifying struggling students based on early-term behavioral indicators.

---

## 📂 Repository Structure
```text
StudentPrediction/
│
├── data/
│   ├── student_data.csv          # Raw generated student dataset
│   └── processed/
│       ├── scaler.joblib         # Fitted StandardScaler object for production deployment
│       └── split_data.npz        # Compressed numpy arrays containing training/testing splits
│
├── models/
│   ├── logistic_regression_model.joblib
│   ├── decision_tree_model.joblib
│   └── random_forest_model.joblib
│
├── results/
│   ├── correlation_matrix.png     # Heatmap visualization
│   ├── feature_distributions.png  # Data distribution histograms
│   ├── model_comparison.csv       # Tabular summary of performance metrics
│   └── model_comparison.png       # Grouped bar chart comparing models
│
├── src/                           # Source folder for clean, modular code
│   ├── generate_data.py           # The data simulator script
│   ├── preprocess.py              # Cleaning, splitting, and scaling logic
│   └── train.py                   # Model training and metric generation
│
└── requirements.txt              # Project environment dependencies
```

## 📊 Dataset Description
The dataset maps 1,000 students across the following core features:
* **Study_Hours**: Continuous variable indicating weekly independent study tracking (2.0 to 20.0 hours).
* **Attendance_Percentage**: Continuous variable indicating lecture attendance rate (60.0% to 100.0%).
* **Previous_Scores**: Baseline proxy metric for fundamental understanding (30.0% to 100.0%).
* **Sleep_Hours**: Average daily sleep duration (5.0 to 9.0 hours).
* **Participation_Level**: Discrete variable indicating active classroom engagement scaled 1 (Low) to 5 (High).
* **Passed (Target)**: Binary label indicating performance split (1 >= 60%, 0 < 60%).

---

## ⚙️ Methodology & Pipeline
The engineering process is split into three main operational blocks to ensure clean, scalable machine learning practices:

1. **Data Generation & Cleaning**: Created a pseudo-random distribution including continuous uniform scales and discrete categorical integers. Introduced controlled random noise to avoid synthetic overfitting and handled missing values using robust **Median Imputation** on feature columns.
2. **Preprocessing**: Isolated features to eliminate data leakage. Split data using a **Stratified 80/20 train-test split** to maintain true class balance. Applied **Z-score Standardization** ensuring features possess a mean of 0 and variance of 1.
3. **Model Selection**: Evaluated a linear baseline (**Logistic Regression**) against non-parametric tree structures (**Decision Trees** and **Random Forests**).

---

## 📈 Results Summary
The models were evaluated comprehensively across unseen testing data. The performance metrics can be found inside `results/model_comparison.csv` and visualized inside `results/model_comparison.png`.

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9050 | 0.9210 | 0.9430 | 0.9319 |
| **Decision Tree** | 0.9450 | 0.9520 | 0.9650 | 0.9585 |
| **Random Forest** | 0.9600 | 0.9660 | 0.9790 | 0.9725 |

### 🔍 Key Insights
* **Random Forest** achieved the highest overall performance across all tracking metrics, benefiting from ensemble variance reduction.
* **Decision Tree** showed strong performance but is structurally susceptible to higher variance if allowed to grow deep without constraints.
* **Logistic Regression** provided an excellent, highly interpretable linear baseline.