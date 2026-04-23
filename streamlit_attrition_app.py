# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:11:13 2026

@author: HP
"""


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("best_attrition_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.stButton>button {
    background-color: #00C9A7;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("👨‍💼 Employee Attrition Prediction System")
st.markdown("### 🚀 AI-Powered HR Dashboard")
st.markdown("---")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 2])

# ---------------- INPUT ----------------
with col1:
    st.subheader("📋 Employee Details")

    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.7)
    projects = st.slider("Number of Projects", 2, 7, 4)
    hours = st.slider("Monthly Hours", 90, 320, 200)
    time_company = st.slider("Years at Company", 1, 10, 3)

    accident = st.selectbox("Work Accident", [0, 1])
    promotion = st.selectbox("Promotion (Last 5 Years)", [0, 1])

    department = st.selectbox("Department", [
        'sales', 'accounting', 'hr', 'technical', 'support',
        'management', 'IT', 'marketing', 'product_mng', 'RandD'
    ])

    salary = st.selectbox("Salary Level", ['low', 'medium', 'high'])

    predict = st.button("🔍 Predict")

# ---------------- OUTPUT ----------------
with col2:
    st.subheader("📊 Prediction Result")

    if predict:

        input_data = pd.DataFrame({
            'satisfaction_level': [satisfaction],
            'last_evaluation': [evaluation],
            'number_project': [projects],
            'average_monthly_hours': [hours],
            'time_spend_company': [time_company],
            'work_accident': [accident],
            'promotion_last_5years': [promotion],
            'department': [department],
            'salary': [salary]
        })

        # preprocess
        input_processed = preprocessor.transform(input_data)

        try:
            input_processed = input_processed.toarray()
        except:
            pass

        # prediction
        pred = model.predict(input_processed)[0]
        prob = model.predict_proba(input_processed)[0][1]

        colA, colB, colC = st.columns(3)

        with colA:
            if pred == 1:
                st.error("🚨 High Risk")
            else:
                st.success("✅ Low Risk")

        with colB:
            st.metric("Probability", f"{prob:.2%}")

        with colC:
            st.metric("Prediction", "LEAVE" if pred == 1 else "STAY")

        st.markdown("---")

        # ---------------- RECOMMENDATION ----------------
        st.subheader("💡 HR Recommendations")

        if pred == 1:
            st.warning("""
            - Increase salary or benefits  
            - Reduce workload  
            - Provide promotion opportunities  
            - Improve employee engagement  
            """)
        else:
            st.success("""
            - Maintain current work environment  
            - Encourage skill development  
            - Monitor performance regularly  
            """)

        # ---------------- INPUT TABLE ----------------
        st.subheader("📋 Input Summary")
        st.dataframe(input_data)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💻 Final Year ML Project | Built with Streamlit")