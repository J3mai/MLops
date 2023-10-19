import streamlit as st
import numpy as np
import pandas as pd
import mlflow.pyfunc
import mlflow.tracking
import warnings


warnings.filterwarnings("ignore")
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
input_method = st.sidebar.radio("Select Input Method", ("Manual Entry", "Upload CSV"))
prediction = 0
if input_method == "Manual Entry":
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


elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        input_data = pd.read_csv(uploaded_file)

        # Show the uploaded data
        st.subheader("Uploaded Data:")
        st.write(input_data)

        # Make predictions
        prediction = model.predict(input_data)

st.subheader("Prediction:")
if prediction == 0:
    st.write("No Diabetes")
else:
    st.write("Diabetes")

# st.sidebar.subheader("Model Metrics:")
# with st.sidebar:
#     st.write("Accuracy Score:", best_run["metric.acurracy"])
#     st.write("Precision Score:", best_run["metric.precision_score"])
#     st.write("Recall Score:", best_run["metric.recall_score"])
#     st.write("F1 Score:", best_run["metric.f1_score"])
