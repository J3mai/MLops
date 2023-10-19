import streamlit as st
import numpy as np
import pandas as pd
import mlflow.pyfunc
import mlflow.tracking

# Connect to the MLflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/J3mai/MlOps.mlflow")

# Retrieve the experiment ID
experiment_id = mlflow.get_experiment_by_name("Default").experiment_id

# Get the best run based on a specific metric
best_run = mlflow.search_runs(
    experiment_ids=[experiment_id], filter_string="", order_by=["metric.f1_score DESC"]
).iloc[0]

# Load the best model
model_uri = (
    best_run.artifact_uri + "/Logreg"
)  # Replace "Logreg" with the actual model name
model = mlflow.pyfunc.load_model(model_uri)

st.title("Diabetes Prediction App")

# Create input fields for user input
pregnancies = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose", 0, 200, 0)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 0)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 0)
insulin = st.number_input("Insulin", 0, 200, 0)
bmi = st.number_input("BMI", 0.0, 60.0, 0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.0, 0.0)
age = st.number_input("Age", 0, 100, 0)

# Make predictions
input_data = pd.DataFrame(
    {
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age],
    }
)
prediction = model.predict(input_data)[0]

st.subheader("Prediction:")
if prediction == 0:
    st.write("No Diabetes")
else:
    st.write("Diabetes")

st.sidebar.subheader("Model Metrics:")

