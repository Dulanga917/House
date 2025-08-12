import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------
# Load model and data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/diabetes.csv")
    return df

@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

df = load_data()
model, scaler = load_model()

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualisation", "Model Prediction", "Model Performance"])

# ---------------------------
# Home Page
# ---------------------------
if menu == "Home":
    st.title("Diabetes Prediction Web App")
    st.markdown("""
    ### Project Description
    This application predicts the likelihood of diabetes in patients based on medical input data.
    - Built with **Streamlit**  
    - Trained on the **Pima Indians Diabetes Dataset**  
    - Model deployed using **Streamlit Cloud**
    """)

# ---------------------------
# Data Exploration
# ---------------------------
elif menu == "Data Exploration":
    st.header("Data Exploration")
    
    st.subheader("Dataset Overview")
    st.write(f"**Shape:** {df.shape}")
    st.write(df.dtypes)
    
    st.subheader("Sample Data")
    st.write(df.head())
    
    st.subheader("Interactive Data Filter")
    col = st.selectbox("Select Column to Filter", df.columns)
    unique_vals = df[col].unique()
    val = st.selectbox("Select Value", unique_vals)
    st.write(df[df[col] == val])

# ---------------------------
# Visualisation Section
# ---------------------------
elif menu == "Visualisation":
    st.header("Data Visualisation")
    
    # Plot 1: Outcome distribution
    fig1 = px.histogram(df, x="Outcome", title="Outcome Distribution")
    st.plotly_chart(fig1)
    
    # Plot 2: Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig2, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig2)
    
    # Plot 3: BMI vs Glucose
    st.subheader("BMI vs Glucose")
    fig3 = px.scatter(df, x="BMI", y="Glucose", color="Outcome", title="BMI vs Glucose by Outcome")
    st.plotly_chart(fig3)

# ---------------------------
# Model Prediction
# ---------------------------
elif menu == "Model Prediction":
    st.header("Predict Diabetes")
    
    st.write("Enter patient information below:")
    
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose", 0, 200, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 140, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 0, 120, 30)
    
    if st.button("Predict"):
        try:
            features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)[0][1]
            
            st.success(f"Prediction: {'Diabetic' if prediction[0]==1 else 'Not Diabetic'}")
            st.info(f"Confidence: {round(probability*100, 2)}%")
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------
# Model Performance
# ---------------------------
elif menu == "Model Performance":
    st.header("Model Performance Metrics")
    
    from sklearn.model_selection import train_test_split
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    st.write(f"**Accuracy:** {acc:.2f}")
    
    st.subheader("Confusion Matrix")
    fig4, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig4)
    
    st.subheader("Classification Report")
    st.write(pd.DataFrame(cr).transpose())
