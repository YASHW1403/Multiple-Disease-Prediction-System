import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle
import os
import requests
import json
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

st.set_page_config(page_title='Multiple Disease Prediction System',layout='wide',page_icon="⚕️")

with open (r'Models\best_diabetes_model.pkl','rb') as f:
    best_model=pickle.load(f)
with open(r'Models\diabscaler.pkl','rb')as f:
    diabscaler=pickle.load(f)

best_heartmodel=pickle.load(open(r"Models\heart.pkl",'rb'))
kidney_model=pickle.load(open(r"Models\kidney_try.pkl",'rb'))

with open(r"Models\parkinsons_model.pkl", "rb") as model_file:
    parkinsons_model = pickle.load(model_file)
with open(r"Models\parkscaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Title and Introduction
st.header("🧑‍⚕️ Multiple Disease Prediction System 🏥")
st.write("🩺 This is a Multiple Disease Prediction System, including Analysis, built using **Machine Learning** and deployed using **StreamLit**. 🤖📊")
st.write("By Yash W ✨❤️")

st.sidebar.image(r"Images\Ml.jpg")
st.markdown(
    """
    <style>
    .main-menu {
        font-size: 20px;
        font-weight: bold;
        color: #FFA500;
        margin-bottom: 15px;
    }
    .main-menu select {
        background-color: #4E5D6C;
        color: #FFFFFF;
        font-size: 18px;
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown('<div class="main-menu">Main Menu 🏠</div>', unsafe_allow_html=True)
main_menu = st.sidebar.selectbox(
    "Select an option 🔽",
    ("Home", "Analysis", "Prediction")
)
selected_disease = st.sidebar.selectbox(
    "Select Disease for Symptoms 🏥",
    ("Symptoms", "Diabetes", "Heart Disease", "Kidney Disease", "Parkinson's Disease")
)
if selected_disease == "Symptoms":
    st.sidebar.write("Select a Disease to view Symptoms")
    st.sidebar.image(r"Images\ml4.jpg")
elif selected_disease == "Diabetes":
    st.sidebar.write("Symptoms of **Diabetes**")
    st.sidebar.image(r"Images\Dsymtons.jpeg")
elif selected_disease == "Heart Disease":
    st.sidebar.write("Symptoms of **Heart Disease**")
    st.sidebar.image(r"Images\Hsys.jpg")
elif selected_disease == "Kidney Disease":
    st.sidebar.write("Symptoms of **Kidney Disease**")
    st.sidebar.image(r"Images\Ksys.jpeg")
elif selected_disease == "Parkinson's Disease":
    st.sidebar.write("Symptoms of **Parkinson's Disease**")
    st.sidebar.image(r"Images\Psys.jpg")

c1,c2,c3,c4=st.columns(4)
if main_menu == "Home":
    st.write("Welcome to the **Multiple Disease Prediction System!** 🏥🤖")
    st.image(r"Images\ml2.jpg",use_container_width=False, width=900)
elif main_menu == "Analysis":
    st.image(r"Images\ana.jpg",use_container_width=True)
    st.subheader("Analysis Options 🔍")
    analysis_option = st.selectbox(
        "Choose a Disease for analysis:",
        ("Select a Disease", "Diabetes", "Heart Disease", "Kidney Disease", "Parkinson's Disease"))

    if analysis_option != "Select a Disease":
        st.write(f"**Selected Disease for Analysis:** {analysis_option}")

        # Load sample data for analysis
        if analysis_option == "Diabetes":
            data = pd.read_csv(r"Data\diabetes.csv")
        elif analysis_option == "Heart Disease":
            data = pd.read_csv(r"Data\heart.csv")
        elif analysis_option == "Kidney Disease":
            data = pd.read_csv(r"Data\kidney_disease.csv")
        elif analysis_option == "Parkinson's Disease":
            data = pd.read_csv(r"Data\parkinsons.csv")

        st.write("📊 Sample Data:")
        st.dataframe(data.head(10))
        st.write("📑 Basic Statistics:")
        st.write(data.describe())
        st.write("📈 Sample Graph:")

#########################################  ANALYSIS PART ########################################################

        if analysis_option == "Diabetes":
            st.subheader("Age Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(data, x="Age", kde=True, bins=30, hue="Outcome", palette="Set1")
            ax.set_title("Age Distribution")
            st.pyplot(fig)

            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

            st.subheader("Glucose vs. BMI")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x="Glucose", y="BMI", hue="Outcome", data=data, palette="Set1", ax=ax)
            ax.set_title("Glucose vs. BMI")
            st.pyplot(fig)

            st.subheader("Blood Pressure by Outcome")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="Outcome", y="BloodPressure", data=data, palette="Set2", ax=ax)
            ax.set_title("Blood Pressure by Outcome")
            ax.set_xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
            ax.set_ylabel("Blood Pressure")
            st.pyplot(fig)

            st.subheader("Count of Diabetic vs. Non-Diabetic Patients")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(x="Outcome", data=data, palette="Set3", ax=ax)
            ax.set_title("Count of Diabetic vs. Non-Diabetic Patients")
            ax.set_xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            st.subheader("Average Glucose Levels by Outcome")
            avg_glucose = data.groupby("Outcome")["Glucose"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x="Outcome", y="Glucose", data=avg_glucose, palette="Set1", ax=ax)
            ax.set_title("Average Glucose Levels by Outcome")
            ax.set_xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
            ax.set_ylabel("Average Glucose Level")
            st.pyplot(fig)

        elif analysis_option == "Heart Disease":
            st.subheader("Heart Disease Target Count")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(x="target", data=data, palette="Set2", ax=ax, hue="target")
            ax.set_title("Heart Disease Target Count")
            ax.set_xlabel("Target (0 = No Heart Disease, 1 = Heart Disease)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            st.subheader("Age Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(data, x="age", kde=True, bins=30, hue="target", palette="Set1", ax=ax)
            ax.set_title("Age Distribution")
            st.pyplot(fig)

            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

            st.subheader("Age vs. Cholesterol")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x="age", y="chol", hue="target", data=data, palette="Set1", ax=ax)
            ax.set_title("Age vs. Cholesterol")
            st.pyplot(fig)

            st.subheader("Resting Blood Pressure by Target")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="target", y="trestbps", data=data, palette="Set2", ax=ax)
            ax.set_title("Resting Blood Pressure by Target")
            ax.set_xlabel("Target (0 = No Heart Disease, 1 = Heart Disease)")
            ax.set_ylabel("Resting Blood Pressure")
            st.pyplot(fig)

            st.subheader("Average Cholesterol by Target")
            avg_chol = data.groupby("target")["chol"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x="target", y="chol", data=avg_chol, palette="Set1", ax=ax)
            ax.set_title("Average Cholesterol by Target")
            ax.set_xlabel("Target (0 = No Heart Disease, 1 = Heart Disease)")
            ax.set_ylabel("Average Cholesterol Level")
            st.pyplot(fig)


        elif analysis_option == "Kidney Disease":
            data["classification"] = data["classification"].str.strip()
            data["classification"] = data["classification"].str.replace(r"[^a-zA-Z0-9]", "", regex=True)

            st.subheader("Kidney Disease Classification Count")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(x="classification", data=data, palette="Set2", ax=ax, hue="classification")
            ax.set_title("Kidney Disease Classification Count")
            st.pyplot(fig)

            st.subheader("Age Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(data, x="age", kde=True, bins=30, hue="classification", palette="Set1", ax=ax)
            ax.set_title("Age Distribution")
            st.pyplot(fig)

            st.subheader("Correlation Heatmap")
            numeric_data = data.select_dtypes(include=['number'])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

            st.subheader("Age vs. Blood Pressure")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x="age", y="bp", hue="classification", data=data, palette="Set1", ax=ax)
            ax.set_title("Age vs. Blood Pressure")
            st.pyplot(fig)

            st.subheader("Average Blood Pressure by Classification")
            avg_bp = data.groupby("classification")["bp"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x="classification", y="bp", data=avg_bp, palette="Set1", ax=ax)
            ax.set_title("Average Blood Pressure by Classification")
            st.pyplot(fig)

        elif analysis_option == "Parkinson's Disease":
            if 'name' in data.columns:
                data = data.drop(columns=['name'])

            st.subheader("Parkinson's Disease Status Count")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x="status", data=data, palette="Set2", ax=ax)
            ax.set_title("Parkinson's Disease Status Count")
            st.pyplot(fig)

            st.subheader("MDVP:Fo(Hz) Distribution by Status")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data=data, x="MDVP:Fo(Hz)", kde=True, bins=30, hue="status", palette="Set1", ax=ax)
            ax.set_title("MDVP:Fo(Hz) Distribution by Status")
            st.pyplot(fig)

            st.subheader("Correlation Heatmap")
            numeric_data = data.select_dtypes(include=['number'])
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax,annot_kws={"size": 8})
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

            st.subheader("MDVP:Fo(Hz) vs. MDVP:Fhi(Hz) by Status")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x="MDVP:Fo(Hz)", y="MDVP:Fhi(Hz)", hue="status", data=data, palette="Set1", ax=ax)
            ax.set_title("MDVP:Fo(Hz) vs. MDVP:Fhi(Hz) by Status")
            st.pyplot(fig)

            st.subheader("MDVP:Flo(Hz) by Status")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x="status", y="MDVP:Flo(Hz)", hue="status", data=data, palette="Set2", ax=ax)
            ax.set_title("MDVP:Flo(Hz) by Status")
            st.pyplot(fig)

            st.subheader("Average MDVP:Shimmer by Status")
            avg_shimmer = data.groupby("status")[["MDVP:Shimmer"]].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="status", y="MDVP:Shimmer", data=avg_shimmer, palette="Set1", ax=ax)
            ax.set_title("Average MDVP:Shimmer by Status")
            st.pyplot(fig)

################################################# Prediction Part ########################################################

elif main_menu == "Prediction":
    c1.image(r"Images\Diab.jpg",caption="Diabetes",use_container_width=True)
    c2.image(r"Images\H2.jpg",caption="Heart",use_container_width=True)
    c3.image(r"Images\K1.jpg",caption="Kidney",use_container_width=True)
    c4.image(r"Images\P2.jpg",caption="Parkinsons",use_container_width=True)
    st.subheader("Prediction Options 🎯")
    prediction_option = st.selectbox(
        "Choose a Disease for prediction:",
        ("Select a Disease", "Diabetes", "Heart Disease", "Kidney Disease", "Parkinson's Disease")
    )

    if prediction_option != "Select a Disease":
        st.write(f"**Selected Disease for Prediction:** {prediction_option}")
 
 ######################## API PART ############################################

    # Load API key from .env
    load_dotenv()
    GROQ_API_KEY = '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'
    # Groq API URL
    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

    # Function to get precautions using Groq API
    def get_precautions(user_data):
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        # Prepare the prompt for the AI
        prompt = f"Based on the following user data, provide precautions for managing disese:\n{user_data}"

        payload = {
            "model": "llama3-8b-8192",  # Update model if needed
            "temperature": 0.7,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(GROQ_URL, json=payload, headers=headers)

        if response.status_code == 200:
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        else:
            return f"API request failed! Status Code: {response.status_code}, Response: {response.text}"

    # Diabetes Prediction Section
    if prediction_option == "Diabetes":
        st.title("Diabetes Prediction 🩺")
        st.write("Please fill in the following details to predict diabetes:")
        c1,c2,c3,c4=st.columns(4)
        pregnancies = c1.number_input("Number of Pregnancies 🤰", min_value=0, max_value=17, value=0)
        glucose = c2.number_input("Glucose Level 🍬", min_value=0, max_value=220, value=0)
        blood_pressure = c3.number_input("Blood Pressure 💓", min_value=0, max_value=122, value=0)
        skin_thickness = c4.number_input("Skin Thickness 🏋️", min_value=0, max_value=99, value=0)
        insulin = c1.number_input("Insulin Level 💉", min_value=0, max_value=846, value=0)
        bmi = c2.number_input("Body Mass Index (BMI) ⚖️", min_value=0.0, max_value=67.1, value=0.0)
        diabetes_pedigree_function = c3.number_input("Diabetes Pedigree Function 🧬", min_value=0.0, max_value=2.42, value=0.0)
        age = c4.number_input("Age 🎂", min_value=0, max_value=81, value=0)

        if st.button("Predict 🔍"):
            input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
            input_data_scaled = diabscaler.transform(input_data)
            prediction = best_model.predict(input_data_scaled)

            if prediction[0] == 1:
                st.error("You are likely to have diabetes. Please consult a healthcare professional for further diagnosis and treatment.")
                
                # Prepare user data for Groq API
                user_data = {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "Blood Pressure": blood_pressure,
                    "Skin Thickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "Diabetes Pedigree Function": diabetes_pedigree_function,
                    "Age": age,
                    'Disease':'Diabetes'
                }

                # Fetch precautions using Groq API
                st.subheader("Precautions for Managing Diabetes")
                precautions = get_precautions(user_data)
                st.write(precautions)
            else:
                st.success("You are not likely to have diabetes. However, it is recommended to maintain a healthy lifestyle to prevent the risk of developing diabetes.")
        
    elif prediction_option == "Heart Disease":
        st.title("Heart Disease Prediction ❤️‍🔥")
        st.write("Please fill in the following details to predict heart disease:")
        c1,c2,c3,c4=st.columns(4)
        age = c1.number_input("Age 🎂", min_value=0, max_value=85, value=0)
        sex = c2.number_input("Gender 🚻", min_value=0, max_value=1, value=0)
        cp = c3.number_input("Chest Pain Type ❤️‍🔥", min_value=0, max_value=3, value=0)
        trestbps = c4.number_input("Resting Blood Pressure 🔴", min_value=0, max_value=200, value=0)
        chol = c1.number_input("Cholesterol Level 🍳", min_value=0, max_value=600, value=0)
        fbs = c2.number_input("Fasting Blood Sugar 🍬", min_value=0, max_value=1, value=0)
        restecg = c3.number_input("Resting ECG 🩺", min_value=0, max_value=3, value=0)
        thalach = c4.number_input("Max Heart Rate 🏃", min_value=0, max_value=200, value=0)
        exang = c1.number_input("Exercise Induced Angina 🏋️", min_value=0, max_value=1, value=0)
        oldpeak = c2.number_input("ST Depression 📉", min_value=0.0, max_value=6.2, value=0.0)
        slope = c3.number_input("Slope 📈", min_value=0, max_value=2, value=0)
        ca = c4.number_input("Major Vessels 🩸", min_value=0, max_value=4, value=0)
        thal = c1.number_input("Thallium Test Result 🔬", min_value=0, max_value=3, value=0)

        if st.button("Predict 🔍"):
            input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
            prediction = best_heartmodel.predict(input_data)

            if prediction[0] == 0:
                st.error("You are likely to have heart disease. Please consult a healthcare professional for further diagnosis and treatment.")

                # Prepare user data for Groq API
                user_data = {
                    "Age": age,
                    "Gender": sex,
                    "Chest Pain Type": cp,
                    "Resting Blood Pressure": trestbps,
                    "Cholesterol Level": chol,
                    "Fasting Blood Sugar": fbs,
                    "Resting ECG": restecg,
                    "Max Heart Rate": thalach,
                    "Exercise Induced Angina": exang,
                    "ST Depression": oldpeak,
                    "Slope": slope,
                    "Major Vessels": ca,
                    "Thallium Test Result": thal,
                    'Disease':'Heart Disease'
                }

                # Fetch precautions using Groq API
                st.subheader("Precautions for Managing Heart Disease")
                precautions = get_precautions(user_data)
                st.write(precautions)
            else:
                st.success("You are not likely to have heart disease. However, it is recommended to maintain a healthy lifestyle to prevent the risk of developing heart disease.")

    elif prediction_option == "Kidney Disease":
        st.title("Kidney Disease Prediction 💧")
        st.write("Please fill in the following details to predict the likelihood of having kidney disease.")
        c1,c2,c3,c4=st.columns(4)
        age = c1.number_input("Age 🎂", min_value=0, max_value=100)
        bp = c2.number_input("Blood Pressure (mmHg) 🩸", min_value=50, max_value=200)
        sg = c3.number_input("Specific Gravity ⚖️", min_value=1.000, max_value=2.000, step=0.001)
        al = c4.number_input("Albumin (g/dL) 🥛", min_value=0, max_value=5)
        su = c1.number_input("Sugar Level 🍬", min_value=0, max_value=5)
        rbc = c2.selectbox("Red Blood Cells 🔴", [0, 1], format_func=lambda x: "Normal" if x == 0 else "Abnormal")
        pc = c3.selectbox("Pus Cells 🦠", [0, 1], format_func=lambda x: "Normal" if x == 0 else "Abnormal")
        pcc = c4.selectbox("Pus Cell Clumps 🧫", [0, 1], format_func=lambda x: "Not Present" if x == 0 else "Present")
        ba = c1.selectbox("Bacteria 🦠", [0, 1], format_func=lambda x: "Not Present" if x == 0 else "Present")
        bgr = c2.number_input("Blood Glucose Random (mg/dL) 🍭", min_value=50, max_value=400)
        bu = c3.number_input("Blood Urea (mg/dL) 🦾", min_value=0, max_value=150, value=0)
        sc = c4.number_input("Serum Creatinine (mg/dL) 🏥", min_value=0.1, max_value=15.0, step=0.1)
        sod = c1.number_input("Sodium (mEq/L) 🧂", min_value=100, max_value=160)
        pot = c2.number_input("Potassium (mEq/L) 🍌", min_value=2.0, max_value=10.0, step=0.1)
        hemo = c3.number_input("Hemoglobin (g/dL) 💉", min_value=3.1, max_value=18.0, step=0.1)
        pcv = c4.number_input("Packed Cell Volume 📦", min_value=10, max_value=55)
        wc = c1.number_input("White Blood Cell Count (cells/cu mm) ⚪", min_value=2000, max_value=20000)
        rc = c2.number_input("Red Blood Cell Count (million cells/cu mm) 🔴", min_value=2.0, max_value=7.0, step=0.1)
        htn = c3.selectbox("Hypertension ⬆️", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        dm = c4.selectbox("Diabetes Mellitus 🚫", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        cad = c1.selectbox("Coronary Artery Disease ❤️", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        appetite = c2.selectbox("Appetite 🍽️", [0, 1], format_func=lambda x: "Good" if x == 0 else "Poor")
        peda_edema = c3.selectbox("Pedal Edema 🦶", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        anaemia = c4.selectbox("Anaemia 🔻", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        if st.button("Predict 🔍"):
            input_data = [[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appetite, peda_edema, anaemia]]
            prediction = kidney_model.predict(input_data)

            if prediction[0] == 0:
                st.error("You are likely to have kidney disease. Please consult a healthcare professional.")

                # Prepare user data for Groq API
                user_data = {
                    "Age": age,
                    "Blood Pressure": bp,
                    "Specific Gravity": sg,
                    "Albumin": al,
                    "Sugar Level": su,
                    "Red Blood Cells": rbc,
                    "Pus Cells": pc,
                    "Pus Cell Clumps": pcc,
                    "Bacteria": ba,
                    "Blood Glucose Random": bgr,
                    "Blood Urea": bu,
                    "Serum Creatinine": sc,
                    "Sodium": sod,
                    "Potassium": pot,
                    "Hemoglobin": hemo,
                    "Packed Cell Volume": pcv,
                    "White Blood Cell Count": wc,
                    "Red Blood Cell Count": rc,
                    "Hypertension": htn,
                    "Diabetes Mellitus": dm,
                    "Coronary Artery Disease": cad,
                    "Appetite": appetite,
                    "Pedal Edema": peda_edema,
                    "Anaemia": anaemia
                }

                # Fetch precautions using Groq API
                st.subheader("Precautions for Managing Kidney Disease")
                precautions = get_precautions(user_data)
                st.write(precautions)
            else:
                st.success("You are not likely to have kidney disease. Maintain a healthy lifestyle to prevent risks.")

    elif prediction_option == "Parkinson's Disease":
        st.title("Parkinson's Disease Prediction 🧠")
        st.write("Please fill in the following details to predict the likelihood of having Parkinson's disease.")
        c1,c2,c3,c4=st.columns(4)
        # Input fields for Parkinson's Disease prediction
        fo = c1.number_input("Fundamental Frequency (MDVP:Fo in Hz) 🎵", min_value=0.0, max_value=300.0, step=0.1, format="%.5f")
        fhi = c2.number_input("Max Frequency (MDVP:Fhi in Hz) 📈", min_value=0.0, max_value=600.0, step=0.1, format="%.5f")
        flo = c3.number_input("Min Frequency (MDVP:Flo in Hz) 📉", min_value=0.0, max_value=300.0, step=0.1, format="%.5f")
        Jitter_percent = c4.number_input("Jitter (%) ⚡", min_value=0.0, max_value=1.0, step=0.001, format="%.5f")
        Jitter_Abs = c1.number_input("Absolute Jitter 🎙️", min_value=0.0, max_value=0.1, step=0.00001, format="%.5f")
        RAP = c2.number_input("Relative Amplitude Perturbation (RAP) 🔊", min_value=0.0, max_value=0.1, step=0.001, format="%.5f")
        PPQ = c3.number_input("Pitch Period Perturbation Quotient (PPQ) 🎶", min_value=0.0, max_value=0.1, step=0.001, format="%.5f")
        DDP = c4.number_input("Jitter:DDP 📊", min_value=0.0, max_value=0.1, step=0.001, format="%.5f")
        Shimmer = c1.number_input("Shimmer 🌊", min_value=0.0, max_value=1.0, step=0.001, format="%.5f")
        Shimmer_dB = c2.number_input("Shimmer (dB) 🔉", min_value=0.0, max_value=10.0, step=0.1, format="%.5f")
        APQ3 = c3.number_input("APQ3 🔍", min_value=0.0, max_value=0.1, step=0.001, format="%.5f")
        APQ5 = c4.number_input("APQ5 📡", min_value=0.0, max_value=0.1, step=0.001, format="%.5f")
        APQ = c1.number_input("Amplitude Perturbation Quotient 📊", min_value=0.0, max_value=0.1, step=0.001, format="%.5f")
        DDA = c2.number_input("Shimmer:DDA 🔬", min_value=0.0, max_value=0.1, step=0.001, format="%.5f")
        NHR = c3.number_input("Noise-to-Harmonics Ratio (NHR) 🔕", min_value=0.0, max_value=1.0, step=0.001, format="%.5f")
        HNR = c4.number_input("Harmonics-to-Noise Ratio (HNR) 🔔", min_value=0.0, max_value=50.0, step=0.1, format="%.5f")
        RPDE = c1.number_input("Recurrence Period Density Entropy (RPDE) 🔄", min_value=0.0, max_value=1.0, step=0.01, format="%.5f")
        DFA = c2.number_input("Detrended Fluctuation Analysis (DFA) 📉", min_value=0.0, max_value=1.0, step=0.01, format="%.5f")
        spread1 = c3.number_input("Spread 1 (Fo Variation) 🎛️", step=0.1, format="%.5f")
        spread2 = c4.number_input("Spread 2 (Fhi and Flo Variation) 🎚️", min_value=0.0, max_value=1.0, step=0.01, format="%.5f")
        D2 = c1.number_input("Correlation Dimension (D2) 🧩", min_value=0.0, max_value=3.0, step=0.1, format="%.5f")
        PPE = c2.number_input("Pitch Period Entropy (PPE) 🎙️", min_value=0.0, max_value=1.0, step=0.01, format="%.5f")

        if st.button("Predict 🔍"):
            # Prepare input data for prediction
            input_data = [[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                        APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
            
            # Scale the input data
            input_data_scaled = scaler.transform(input_data)
            
            # Predict using the model
            proba = parkinsons_model.predict_proba(input_data_scaled)[:, 1]
            custom_threshold = 0.6  # Adjust threshold as needed
            prediction = (proba > custom_threshold).astype(int)

            if prediction[0] == 1:
                st.error("You are likely to have Parkinson's disease. Please consult a healthcare professional.")

                # Prepare user data for Groq API
                user_data = {
                    "Fundamental Frequency (MDVP:Fo in Hz)": fo,
                    "Max Frequency (MDVP:Fhi in Hz)": fhi,
                    "Min Frequency (MDVP:Flo in Hz)": flo,
                    "Jitter (%)": Jitter_percent,
                    "Absolute Jitter": Jitter_Abs,
                    "Relative Amplitude Perturbation (RAP)": RAP,
                    "Pitch Period Perturbation Quotient (PPQ)": PPQ,
                    "Jitter:DDP": DDP,
                    "Shimmer": Shimmer,
                    "Shimmer (dB)": Shimmer_dB,
                    "APQ3": APQ3,
                    "APQ5": APQ5,
                    "Amplitude Perturbation Quotient": APQ,
                    "Shimmer:DDA": DDA,
                    "Noise-to-Harmonics Ratio (NHR)": NHR,
                    "Harmonics-to-Noise Ratio (HNR)": HNR,
                    "Recurrence Period Density Entropy (RPDE)": RPDE,
                    "Detrended Fluctuation Analysis (DFA)": DFA,
                    "Spread 1 (Fo Variation)": spread1,
                    "Spread 2 (Fhi and Flo Variation)": spread2,
                    "Correlation Dimension (D2)": D2,
                    "Pitch Period Entropy (PPE)": PPE,
                    'Disease':'Parkinsons Disease'
                }

                # Fetch precautions using Groq API
                st.subheader("Precautions for Managing Parkinson's Disease")
                precautions = get_precautions(user_data)
                st.write(precautions)
            else:
                st.success("You are not likely to have Parkinson's disease. Maintain a healthy lifestyle.")