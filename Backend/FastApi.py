from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import os
import pandas as pd

os.environ["MLFLOW_TRACKING_USERNAME"] = "J3mai"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "be902b6737c75750f03c765f32e58f691ff049f0"

# setup mlflow
mlflow.set_tracking_uri(
    "https://dagshub.com/J3mai/MlOps.mlflow"
)  # your mlfow tracking uri
mlflow.set_experiment("default")

all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
df_mlflow = mlflow.search_runs(
    experiment_ids=all_experiments, filter_string="metrics.accuracy <1"
)
best_run = mlflow.search_runs(experiment_ids=[all_experiments], filter_string="",order_by=["metric.acurracy DESC"]).iloc[0]

#run_id = df_mlflow.loc[df_mlflow['metrics.accuracy'].idxmax()]['run_id']
logged_model = f'runs:/{best_run}/ML_models'

model = mlflow.pyfunc.load_model(logged_model)

app = FastAPI()
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"Hello": "Diabetes detector app"}

@app.post("/predict/csv")
def return_predictions(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    preprocessed_data = data
    predictions = model.predict(preprocessed_data)
    return {"predictions": predictions.tolist()}

@app.post("/predict")
def predict(data : TransactionModel):
    received = data.dict()
    df =  pd.DataFrame(received,index=[0])
    preprocessed_data = clean_data_json(df)
    predictions = model.predict(preprocessed_data)
    return {"predictions": predictions.tolist()}