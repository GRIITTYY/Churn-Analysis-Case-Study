# 📊 Customer Churn Analysis Case Study

A complete end-to-end customer churn analysis project for a telecommunications company. This project covers data preprocessing, exploratory data analysis (EDA), machine learning modeling, and model interpretability using the Telco Customer Churn dataset.

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telecomchurnpredictor. streamlit.app)

**Predict customer churn risk instantly with our interactive ML-powered app!**

<a href="https://telecomchurnpredictor.streamlit.app">
  <img src="https://img.shields.io/badge/🔮_Live_Demo-Try_Now! -FF4B4B? style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"/>
</a>

</div>

---

## 🎯 Project Overview

### What is Churn? 
A customer is considered **churned** if they discontinue their telecom subscription (Churn = Yes).

### Why Churn Matters
Understanding customer churn is crucial for businesses to:
- Identify patterns, factors, and indicators that contribute to revenue loss
- Reduce customer acquisition costs (acquiring new customers is more expensive than retaining existing ones)
- Enable targeted retention strategies by identifying high-risk customers

## 📁 Project Structure

```
Churn-Analysis-Case-Study/
├── 1.  Customer Churn Dataset Preprocessing. ipynb   # Data cleaning & preprocessing
├── 2. Customer Churn Dataset EDA.ipynb             # Exploratory data analysis
├── 3. Customer_Churn_Dataset_ML_Modeling.ipynb     # ML model training & evaluation
├── churn_app.py                                     # Streamlit app (WIP)
├── requirements.txt                                 # Python dependencies
├── dataset/                                         # Raw and processed datasets
│   ├── WA_Fn-UseC_-Telco-Customer-Churn 2.csv
│   ├── Part1-Telco-Customer-Churn. csv
│   └── Part2-Telco-Customer-Churn. csv
├── model/                                           # Saved ML model
│   └── gradient_boosting_churn_model.pkl
└── plots/                                           # Generated visualizations
    └── univariate_churn_subplots.png
```

## 📊 Dataset

The dataset contains **7,043 customers** with **21 features** including: 

| Feature | Description |
|---------|-------------|
| customerID | Unique customer identifier |
| gender | Male or Female |
| SeniorCitizen | Whether customer is 65+ years old |
| Partner | Whether customer has a partner |
| Dependents | Whether customer has dependents |
| tenure | Months with the company |
| PhoneService | Whether customer has phone service |
| MultipleLines | Whether customer has multiple phone lines |
| InternetService | Type of internet (DSL, Fiber optic, No) |
| OnlineSecurity | Whether customer has online security |
| OnlineBackup | Whether customer has online backup |
| DeviceProtection | Whether customer has device protection |
| TechSupport | Whether customer has tech support |
| StreamingTV | Whether customer streams TV |
| StreamingMovies | Whether customer streams movies |
| Contract | Contract type (Month-to-month, One year, Two year) |
| PaperlessBilling | Whether customer uses paperless billing |
| PaymentMethod | Payment method used |
| MonthlyCharges | Monthly charge amount |
| TotalCharges | Total amount charged |
| Churn | Whether customer churned (Target variable) |

## 🔍 Key Insights from EDA

### Churn Rate
- **26.5%** of customers have churned
- Imbalanced dataset with more non-churners than churners

### Key Churn Indicators

**Demographics:**
- Senior citizens are more likely to churn
- Customers without partners or dependents are more likely to churn

**Services:**
- Fiber optic internet users are more likely to churn
- Customers without OnlineSecurity, OnlineBackup, DeviceProtection, or TechSupport are more likely to churn

**Contract & Billing:**
- Month-to-month contract customers are significantly more likely to churn
- Paperless billing customers show higher churn rates
- Electronic check payment method is associated with higher churn

**Tenure & Charges:**
- ~50% of churned customers leave within the first 10 months
- Churned customers tend to have higher MonthlyCharges (~$15 more on average)
- Churn is prevalent in the $70–$100/month pricing tier

## 🤖 Machine Learning Models

The project evaluates multiple classification algorithms:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier** ✅ (Best performing)
- **XGBoost Classifier**
- **CatBoost Classifier**

### Feature Engineering Pipeline
- Data type coercion for numerical columns
- Custom imputation for missing TotalCharges values
- StandardScaler for numerical features
- OneHotEncoder for categorical features

### Model Evaluation Metrics
- Classification Report (Precision, Recall, F1-Score)
- ROC-AUC Score
- Confusion Matrix
- SHAP values for model interpretability

---

### 🚀 [Try the Live Demo →](https://telecomchurnpredictor.streamlit.app)

----

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone https://github.com/GRIITTYY/Churn-Analysis-Case-Study.git
cd Churn-Analysis-Case-Study
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebooks in order:
   - Start with `1. Customer Churn Dataset Preprocessing.ipynb`
   - Then `2. Customer Churn Dataset EDA.ipynb`
   - Finally `3. Customer_Churn_Dataset_ML_Modeling.ipynb`

## 📦 Dependencies

```
pandas==2.3.1
streamlit==1.52.2
plotly==6.5.0
seaborn==0.13.2
matplotlib==3.10.0
imbalanced-learn==0.14.1
scikit-learn==1.7.1
shap==0.50.0
xgboost==3.1.2
catboost==1.2.8
```

## 📈 Business Recommendations

Based on the analysis: 

1. **Focus on early retention**: Target customers in their first 12 months with loyalty programs
2. **Review fiber optic service quality**: Investigate why fiber optic customers churn more
3. **Promote longer contracts**: Offer incentives for one-year or two-year contracts
4. **Bundle protection services**: Encourage adoption of OnlineSecurity, TechSupport, etc.
5. **Address payment friction**: Electronic check users show higher churn—consider promoting automatic payments

## 🔮 Future Enhancements
- [ ] Add customer segmentation analysis
- [ ] Implement model monitoring and drift detection
- [ ] Create automated retraining pipeline

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/GRIITTYY/Churn-Analysis-Case-Study/issues).

---

**Author:** [GRIITTYY](https://github.com/GRIITTYY)