# =========================================================
# CUSTOMER CHURN PREDICTION DASHBOARD
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================

import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")

# =========================================================
# LOAD DATA
# =========================================================

data = pd.read_csv(
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# =========================================================
# DATA CLEANING
# =========================================================

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

data.dropna(inplace=True)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

data["ChargesPerMonth"] = (
    data["TotalCharges"] /
    (data["tenure"] + 1)
)

data["SeniorWithNoSupport"] = (
    (data["SeniorCitizen"] == 1) &
    (data["TechSupport"] == "No")
).astype(int)

# =========================================================
# TARGET ENCODING
# =========================================================

data["ChurnEncoded"] = data["Churn"].map({
    "Yes":1,
    "No":0
})

# =========================================================
# FEATURES & TARGET
# =========================================================

X = data.drop(
    ["customerID", "Churn", "ChurnEncoded"],
    axis=1
)

y = data["ChurnEncoded"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================================================
# PREPROCESSING
# =========================================================

categorical_cols = X.select_dtypes(
    include="object"
).columns

numeric_cols = X.select_dtypes(
    include=np.number
).columns

preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            StandardScaler(),
            numeric_cols
        ),

        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ]
)

# =========================================================
# MODELS
# =========================================================

lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     LogisticRegression(max_iter=1000))
])

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     RandomForestClassifier(
         random_state=42
     ))
])

gb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     GradientBoostingClassifier())
])

# =========================================================
# TRAIN MODELS
# =========================================================

lr_model.fit(X_train, y_train)

rf_model.fit(X_train, y_train)

gb_model.fit(X_train, y_train)

# =========================================================
# PREDICTIONS
# =========================================================

lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:,1]

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:,1]

gb_pred = gb_model.predict(X_test)
gb_prob = gb_model.predict_proba(X_test)[:,1]

# =========================================================
# METRICS FUNCTION
# =========================================================

def model_metrics(y_test, pred, prob):

    return {
        "Accuracy":
        accuracy_score(y_test, pred),

        "Precision":
        precision_score(y_test, pred),

        "Recall":
        recall_score(y_test, pred),

        "F1 Score":
        f1_score(y_test, pred),

        "ROC-AUC":
        roc_auc_score(y_test, prob)
    }

# =========================================================
# MODEL RESULTS
# =========================================================

lr_metrics = model_metrics(
    y_test,
    lr_pred,
    lr_prob
)

rf_metrics = model_metrics(
    y_test,
    rf_pred,
    rf_prob
)

gb_metrics = model_metrics(
    y_test,
    gb_pred,
    gb_prob
)

# =========================================================
# TOP METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    len(data)
)

col2.metric(
    "Churn Customers",
    len(data[data["Churn"] == "Yes"])
)

col3.metric(
    "Best Accuracy",
    round(
        max(
            lr_metrics["Accuracy"],
            rf_metrics["Accuracy"],
            gb_metrics["Accuracy"]
        ) * 100,
        2
    )
)

col4.metric(
    "Best ROC-AUC",
    round(
        max(
            lr_metrics["ROC-AUC"],
            rf_metrics["ROC-AUC"],
            gb_metrics["ROC-AUC"]
        ),
        3
    )
)

# =========================================================
# DATA PREVIEW
# =========================================================

with st.expander("📁 View Dataset"):

    st.write(data.head())

# =========================================================
# MODEL PERFORMANCE TABLE
# =========================================================

st.subheader("📈 Model Performance")

metrics_df = pd.DataFrame({

    "Logistic Regression":
    lr_metrics,

    "Random Forest":
    rf_metrics,

    "Gradient Boosting":
    gb_metrics
})

st.dataframe(metrics_df)

# =========================================================
# RISK SEGMENTATION
# =========================================================

risk = []

for p in rf_prob:

    if p >= 0.70:

        risk.append("High Risk")

    elif p >= 0.40:

        risk.append("Medium Risk")

    else:

        risk.append("Low Risk")

risk_df = pd.DataFrame({

    "RiskTier": risk
})

# =========================================================
# SMALL DASHBOARD CHARTS
# =========================================================

st.subheader("📊 Dashboard Charts")

# =========================================================
# ROW 1
# =========================================================

col5, col6 = st.columns(2)

# -------------------------
# CHURN DISTRIBUTION
# -------------------------

with col5:

    st.write("### Churn Distribution")

    fig1, ax1 = plt.subplots(figsize=(4,3))

    sns.countplot(
        x="Churn",
        data=data,
        palette={
            "Yes":"green",
            "No":"red"
        },
        ax=ax1
    )

    st.pyplot(fig1)

# -------------------------
# CONTRACT TYPE
# -------------------------

with col6:

    st.write("### Contract Type")

    contract_df = data.groupby(
        "Contract"
    )["ChurnEncoded"].mean().reset_index()

    fig2, ax2 = plt.subplots(figsize=(4,3))

    sns.barplot(
        x="Contract",
        y="ChurnEncoded",
        data=contract_df,
        palette=[
            "red",
            "orange",
            "green"
        ],
        ax=ax2
    )

    ax2.set_title("Churn by Contract Type")

    st.pyplot(fig2)

# =========================================================
# ROW 2
# =========================================================

col7, col8 = st.columns(2)

# -------------------------
# TENURE DISTRIBUTION
# -------------------------

with col7:

    st.write("### Tenure Distribution")

    fig3, ax3 = plt.subplots(figsize=(4,3))

    sns.histplot(
        data=data,
        x="tenure",
        hue="Churn",
        kde=True,
        ax=ax3
    )

    st.pyplot(fig3)

# -------------------------
# RISK SEGMENTATION
# -------------------------

with col8:

    st.write("### Risk Segmentation")

    fig4, ax4 = plt.subplots(figsize=(4,3))

    risk_df["RiskTier"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax4
    )

    st.pyplot(fig4)

# =========================================================
# LARGE CHART VIEW
# =========================================================

st.subheader("🔍 Detailed Large Charts")

chart_option = st.selectbox(

    "Select Chart",

    [
        "Churn Distribution",
        "Contract Type",
        "Tenure Distribution",
        "Risk Segmentation",
        "ROC Curve",
        "Feature Importance",
        "Confusion Matrix",
        "Correlation Heatmap"
    ]
)

# =========================================================
# CHURN DISTRIBUTION LARGE
# =========================================================

if chart_option == "Churn Distribution":

    fig, ax = plt.subplots(figsize=(10,6))

    sns.countplot(
        x="Churn",
        data=data,
        palette={
            "Yes":"green",
            "No":"red"
        },
        ax=ax
    )

    st.pyplot(fig)

# =========================================================
# CONTRACT TYPE LARGE
# =========================================================

elif chart_option == "Contract Type":

    fig, ax = plt.subplots(figsize=(10,6))

    sns.barplot(
        x="Contract",
        y="ChurnEncoded",
        data=contract_df,
        palette=[
            "red",
            "orange",
            "green"
        ],
        ax=ax
    )

    ax.set_title("Churn by Contract Type")

    st.pyplot(fig)

# =========================================================
# TENURE DISTRIBUTION LARGE
# =========================================================

elif chart_option == "Tenure Distribution":

    fig, ax = plt.subplots(figsize=(10,6))

    sns.histplot(
        data=data,
        x="tenure",
        hue="Churn",
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

# =========================================================
# RISK SEGMENTATION LARGE
# =========================================================

elif chart_option == "Risk Segmentation":

    fig, ax = plt.subplots(figsize=(10,6))

    risk_df["RiskTier"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    st.pyplot(fig)

# =========================================================
# ROC CURVE LARGE
# =========================================================

elif chart_option == "ROC Curve":

    lr_fpr, lr_tpr, _ = roc_curve(
        y_test,
        lr_prob
    )

    rf_fpr, rf_tpr, _ = roc_curve(
        y_test,
        rf_prob
    )

    gb_fpr, gb_tpr, _ = roc_curve(
        y_test,
        gb_prob
    )

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(
        lr_fpr,
        lr_tpr,
        label="Logistic Regression"
    )

    ax.plot(
        rf_fpr,
        rf_tpr,
        label="Random Forest"
    )

    ax.plot(
        gb_fpr,
        gb_tpr,
        label="Gradient Boosting"
    )

    ax.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    ax.legend()

    ax.set_title("ROC Curve")

    st.pyplot(fig)

# =========================================================
# FEATURE IMPORTANCE LARGE
# =========================================================

elif chart_option == "Feature Importance":

    rf_classifier = rf_model.named_steps[
        "classifier"
    ]

    feature_names = rf_model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance":
        rf_classifier.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    fig, ax = plt.subplots(figsize=(10,6))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        ax=ax
    )

    st.pyplot(fig)

# =========================================================
# CONFUSION MATRIX LARGE
# =========================================================

elif chart_option == "Confusion Matrix":

    cm = confusion_matrix(
        y_test,
        rf_pred
    )

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

# =========================================================
# CORRELATION HEATMAP LARGE
# =========================================================

elif chart_option == "Correlation Heatmap":

    numeric_df = data.select_dtypes(
        include=np.number
    )

    fig, ax = plt.subplots(figsize=(12,8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# =========================================================
# BUSINESS INSIGHTS
# =========================================================

st.subheader("💡 Business Insights")

st.write("""

### Key Findings

1. Month-to-month customers have highest churn.

2. Customers with low tenure are high risk.

3. High monthly charges increase churn probability.

4. Random Forest and Gradient Boosting performed best.

5. Customers without tech support are more likely to churn.

### Recommendations

- Offer yearly contract discounts.
- Improve onboarding support for new customers.
- Provide retention offers for high-risk customers.

""")
