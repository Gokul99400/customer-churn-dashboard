# 📊 Customer Churn Prediction & Risk Segmentation Dashboard

## 📌 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. This project focuses on predicting customer churn using Machine Learning and visualizing customer risk levels through an interactive Streamlit dashboard.

The system analyzes telecom customer data, identifies customers likely to leave the service, and segments them into different risk categories such as High Risk, Medium Risk, and Low Risk.

---

## 🎯 Objectives

- Predict customer churn using Machine Learning models
- Compare multiple classification algorithms
- Analyze important churn-driving factors
- Segment customers based on churn probability
- Build an interactive business dashboard using Streamlit

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🤖 Machine Learning Models Used

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

---

## 📈 Dashboard Features

- Customer churn distribution analysis
- Contract type churn comparison
- Tenure distribution visualization
- Risk segmentation pie chart
- ROC Curve comparison
- Feature importance analysis
- Correlation heatmap
- Confusion matrix visualization

---

## 🔍 Risk Segmentation

Customers are classified into 3 categories based on churn probability:

| Risk Level | Probability |
|---|---|
| 🔴 High Risk | ≥ 0.70 |
| 🟡 Medium Risk | 0.40 – 0.69 |
| 🟢 Low Risk | < 0.40 |

---

## 💡 Key Insights

- Month-to-month contract customers have the highest churn rate.
- Customers with low tenure are more likely to churn.
- High monthly charges increase churn probability.
- Customers without technical support are more likely to leave.

---

## 🚀 Business Recommendations

- Offer discounts for yearly contracts.
- Improve onboarding support for new customers.
- Provide retention offers for high-risk customers.
- Improve customer support services.

---

## 📂 Project Structure

```text
customer-churn-dashboard
│
├── app.py
├── requirements.txt
├── README.md
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## ▶️ How to Run the Project

### Step 1 — Install Libraries

```bash
pip install -r requirements.txt
```

### Step 2 — Run Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Dataset

Dataset used:
Telco Customer Churn Dataset

Source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## 👨‍💻 Author

**S. Gokul Kumar**

AI & ML Student  
Sri Venkateswaraa College of Technology

LinkedIn:
https://www.linkedin.com/in/s-gokul-kumar-147885281
