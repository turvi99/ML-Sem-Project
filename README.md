<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-Ensemble-189FDD?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Accuracy-97.38%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-7043%20Customers-blueviolet?style=for-the-badge" />
</p>

<h1 align="center">📞 Telco Customer Churn Prediction</h1>

<p align="center">
  <strong>An end-to-end Machine Learning pipeline that predicts customer churn for a telecom company — from exploratory data analysis and model comparison to a deployed interactive Streamlit dashboard for real-time predictions.</strong>
</p>

<p align="center">
  <em>Academic Project — Machine Learning (Semester Project)</em>
</p>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Highlights](#-highlights)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
- [Model Comparison](#-model-comparison)
- [SMOTEENN — Handling Class Imbalance](#-smoteenn--handling-class-imbalance)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [Key Findings](#-key-findings)

---

## 🎯 Problem Statement

Customer churn is one of the most critical challenges in the telecom industry. Acquiring new customers costs **5–25x more** than retaining existing ones. This project builds a machine learning model to **predict which customers are likely to churn**, enabling proactive retention strategies.

**Goal**: Given a customer's demographics, account information, and service subscriptions, predict whether they will churn (leave the company).

---

## ✨ Highlights

| Metric | Value |
|:--|:--|
| **Best Model** | K-Nearest Neighbors (with SMOTEENN) |
| **Best Accuracy** | **97.38%** (after class balancing) |
| **Baseline Accuracy** | 78.96% (XGBoost, before balancing) |
| **Models Compared** | 5 (Random Forest, KNN, AdaBoost, XGBoost, Naive Bayes) |
| **Dataset Size** | 7,043 customers × 21 features |
| **Hyperparameter Tuning** | GridSearchCV with Pipeline |
| **Class Imbalance Fix** | SMOTEENN (SMOTE + Edited Nearest Neighbors) |
| **Deployment** | Interactive Streamlit Dashboard |

---

## 📊 Dataset

**Source**: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM Sample Dataset)

| Property | Detail |
|:--|:--|
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target Variable** | `Churn` (Yes/No) |
| **Missing Values** | 11 rows in `TotalCharges` (dropped) |

### Feature Categories

<table>
<tr>
<td>

**Demographics**
- `gender` — Male / Female
- `SeniorCitizen` — 0 / 1
- `Partner` — Yes / No
- `Dependents` — Yes / No

</td>
<td>

**Account Info**
- `tenure` — Months with company
- `Contract` — Month-to-month / 1yr / 2yr
- `PaperlessBilling` — Yes / No
- `PaymentMethod` — 4 options
- `MonthlyCharges` — Dollar amount
- `TotalCharges` — Dollar amount

</td>
<td>

**Services Subscribed**
- `PhoneService`
- `MultipleLines`
- `InternetService` (DSL / Fiber / No)
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`

</td>
</tr>
</table>

---

## 🧠 ML Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Data        │───▶│  Data        │───▶│  Feature     │───▶│  Model       │───▶│  Deploy      │
│  Loading     │    │  Cleaning    │    │  Engineering │    │  Training    │    │  (Streamlit) │
│              │    │              │    │              │    │              │    │              │
│ • Kaggle API │    │ • Drop NaN   │    │ • Label      │    │ • Train/Test │    │ • Predict    │
│ • 7043 rows  │    │ • Drop ID    │    │   Encoding   │    │   80/20 split│    │ • Explore    │
│ • 21 columns │    │ • Fix types  │    │ • Min-Max    │    │ • 5 models   │    │ • Visualize  │
│              │    │              │    │   Normalize  │    │ • GridSearch │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    │ • SMOTEENN   │    └──────────────┘
                                                            └──────────────┘
```

### Pipeline Steps

| Step | Description | Details |
|:-----|:------------|:--------|
| **1. Data Loading** | Load dataset from Kaggle | 7,043 rows × 21 columns |
| **2. Data Cleaning** | Handle missing & invalid data | Convert `TotalCharges` to numeric, drop 11 NaN rows, drop `customerID` |
| **3. EDA** | Exploratory Data Analysis | Count plots, histograms, KDE density plots, correlation heatmap |
| **4. Feature Encoding** | Convert categorical → numerical | Manual label encoding for all 16 categorical features |
| **5. Normalization** | Scale all features to [0,1] | Min-Max normalization applied to every column |
| **6. Train-Test Split** | Partition data | 80% train / 20% test (`random_state=42`) |
| **7. Model Training** | Train & compare 5 models | RandomForest, KNN, AdaBoost, XGBoost, Naive Bayes |
| **8. Hyperparameter Tuning** | Optimize each model | `GridSearchCV` with `sklearn.Pipeline` |
| **9. Class Balancing** | Address churn imbalance | SMOTEENN (SMOTE oversampling + ENN cleaning) |
| **10. Model Export** | Serialize best model | `pickle.dump()` → `model.pkl` |

---

## 📈 Model Comparison

### Before SMOTEENN (Imbalanced Data)

| Rank | Model | Accuracy |
|:----:|:------|:---------|
| 1 | **XGBoost** | 78.96% |
| 2 | Random Forest | 78.75% |
| 3 | AdaBoost | 78.39% |
| 4 | K-Nearest Neighbors | 75.62% |
| 5 | Naive Bayes | 72.78% |

### Initial Random Forest Metrics

| Metric | Score |
|:-------|:------|
| Accuracy | 79% |
| Precision | 78% |
| Recall | 79% |
| F1 Score | 78% |

---

## ⚖️ SMOTEENN — Handling Class Imbalance

The raw dataset has a **class imbalance** — far more "No Churn" than "Yes Churn" customers. This biases models toward predicting "No Churn". 

**Solution**: [SMOTEENN](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html) — a hybrid technique combining:
- **SMOTE** (Synthetic Minority Oversampling) — generates synthetic samples for the minority class
- **ENN** (Edited Nearest Neighbors) — removes noisy/borderline samples from both classes

### After SMOTEENN (Balanced Data)

| Rank | Model | Accuracy | Improvement |
|:----:|:------|:---------|:------------|
| 🥇 | **K-Nearest Neighbors** | **97.38%** | +21.76% |
| 🥈 | Random Forest | 96.35% | +17.60% |
| 🥉 | XGBoost | 96.03% | +17.07% |
| 4 | AdaBoost | 92.85% | +14.46% |
| 5 | Naive Bayes | 89.67% | +16.89% |

> **Best model (KNN at 97.38%) was saved as `model.pkl` and deployed in the Streamlit dashboard.**

---

## 🖥️ Streamlit Dashboard

The project includes a **3-page interactive web dashboard** built with Streamlit:

### 🔮 Page 1: Predict Churn
Fill in a customer's details across **18 input features** and get an instant churn prediction:
- Categorical dropdowns for services (Internet, Phone, Streaming, etc.)
- Numerical inputs for tenure, monthly charges, total charges
- Color-coded result: 🚨 **Likely to Churn** (red) or 💚 **Will NOT Churn** (green)

### 📊 Page 2: Dataset Explorer
Upload your own CSV to explore:
- Interactive data preview table
- Summary statistics (describe)

### 📈 Page 3: Insights Dashboard
Upload your dataset to see interactive Plotly visualizations:
- **Monthly Charges Distribution** — Histogram
- **Gender Breakdown** — Pie chart
- **Churn by Contract Type** — Bar chart (Month-to-month vs. 1yr vs. 2yr)

---

## 📁 Project Structure

```
ML-Sem-Project/
├── Telco_Customer_Churn_ml.ipynb   # 📓 Complete ML pipeline (73 cells)
│                                    #    EDA → Encoding → Normalization →
│                                    #    Model Comparison → SMOTEENN → Export
│
├── app.py                           # 🖥️  Streamlit dashboard (3 pages)
│                                    #    Predict Churn | Dataset Explorer | Insights
│
├── WA_Fn-UseC_-Telco-Customer-     # 📊 Kaggle Telco dataset
│   Churn.csv                        #    7,043 customers × 21 features
│
├── model.pkl                        # 💾 Trained KNN model (SMOTEENN, 97.38%)
│
└── README.md                        # 📖 This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/turvi99/ML-Sem-Project.git
cd ML-Sem-Project

# 2. Install dependencies
pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn plotly matplotlib seaborn

# 3. Launch the Streamlit dashboard
streamlit run app.py
```

### Retrain the Model (Optional)

Open `Telco_Customer_Churn_ml.ipynb` in Jupyter Notebook or Google Colab and run all cells. The notebook will:
1. Download the dataset from Kaggle
2. Perform EDA and preprocessing
3. Train and compare 5 models
4. Apply SMOTEENN for class balancing
5. Export the best model as `model.pkl`

---

## 🛠️ Tech Stack

| Category | Technology |
|:---------|:-----------|
| **Language** | Python 3.8+ |
| **ML Framework** | Scikit-Learn, XGBoost |
| **Class Balancing** | imbalanced-learn (SMOTEENN) |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web Dashboard** | Streamlit |
| **Model Serialization** | pickle |
| **Development** | Google Colab, Jupyter Notebook |
| **Hyperparameter Tuning** | GridSearchCV, RandomizedSearchCV |

---

## 🔍 Key Findings

### From Exploratory Data Analysis

| Insight | Detail |
|:--------|:-------|
| **Contract Type** is the strongest churn predictor | Month-to-month customers churn significantly more than 1yr/2yr contracts |
| **Fiber Optic** internet users churn more | Higher monthly charges + potential service issues drive churn |
| **Tenure** inversely correlates with churn | New customers (low tenure) are most at risk |
| **Monthly Charges** positively correlates with churn | Higher-paying customers are more likely to leave |
| **No Online Security / Tech Support** increases churn | Customers without add-on services churn more |
| **Electronic check** payment correlates with churn | Automated payments (bank/credit card) correlate with retention |

### From Model Analysis

| Insight | Detail |
|:--------|:-------|
| **Class imbalance matters** | SMOTEENN improved accuracy by ~18–22% across all models |
| **KNN outperforms after balancing** | Nearest-neighbor approach benefits most from synthetic samples |
| **Ensemble methods are strong baselines** | RF, XGBoost, and AdaBoost all perform well before and after balancing |
| **Naive Bayes is the weakest** | Feature independence assumption doesn't hold for correlated telecom features |

---

## 📊 Visualizations Performed

The notebook includes **15+ visualizations**:

- 📊 **Count plots** — All categorical features vs. Churn
- 📉 **Histograms** — Tenure, MonthlyCharges, TotalCharges by Churn
- 📈 **KDE Density Plots** — Monthly & Total Charges distributions
- 🔥 **Correlation Heatmap** — All features (annotated, high-resolution)
- 🏅 **Model Comparison Bar Charts** — Before and after SMOTEENN
- 🔲 **Confusion Matrix Heatmap** — Best model performance
- 🟢 **Null Value Heatmap** — Data quality check

---

<p align="center">
  <strong>Built with ❤️ for smarter customer retention</strong><br/>
  <sub>If you found this project useful, consider giving it a ⭐</sub>
</p>
