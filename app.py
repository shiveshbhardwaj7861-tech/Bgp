
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Bhagalpuri Intelligence", layout="wide")

st.title("Bhagalpuri Data Intelligence Dashboard 🚀")

uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Encoding
    df_encoded = df.copy()
    le_dict = {}
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            le_dict[col] = le

    X = df_encoded.drop("purchase_intent", axis=1)
    y = df_encoded["purchase_intent"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ================= CLASSIFICATION =================
    st.header("📊 Classification Model")

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", round(acc,3))
    col2.metric("Precision", round(prec,3))
    col3.metric("Recall", round(rec,3))
    col4.metric("F1 Score", round(f1,3))

    # ROC Curve (One-vs-Rest simplified)
    try:
        y_prob = clf.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob[:,1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        st.pyplot(fig)
    except:
        st.warning("ROC curve could not be plotted (multi-class limitation).")

    # Feature Importance
    st.subheader("Feature Importance")
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    fig2 = plt.figure()
    feat_imp.head(10).plot(kind='bar')
    plt.title("Top Features")
    st.pyplot(fig2)

    # ================= CLUSTERING =================
    st.header("🧠 Customer Segmentation")

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['cluster'] = clusters

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    fig3 = plt.figure()
    plt.scatter(reduced[:,0], reduced[:,1], c=clusters)
    plt.title("Customer Segments (PCA View)")
    st.pyplot(fig3)

    st.write(df['cluster'].value_counts())

    # ================= REGRESSION =================
    st.header("💰 Spending Prediction")

    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train, y_train)
    reg_pred = reg.predict(X_test)

    st.write("Sample Predictions:", reg_pred[:5])

    # ================= ASSOCIATION =================
    st.header("🔗 Association Rules")

    df_bool = df.select_dtypes(include=[int])
    freq_items = apriori(df_bool, min_support=0.1, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1)

    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']].head())

    # ================= NEW CUSTOMER PREDICTION =================
    st.header("🎯 Predict New Customer")

    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"{col}", value=0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        pred = clf.predict(input_df)[0]
        prob = clf.predict_proba(input_df).max()

        st.success(f"Prediction: {pred}")
        st.info(f"Confidence: {round(prob,2)}")
