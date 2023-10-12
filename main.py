from urllib.parse import urlparse
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import matplotlib.pyplot as plt  # side-stepping mpl backend
import matplotlib.gridspec as gridspec  # subplots
import mpld3 as mpl

# Import models from scikit learn module:
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn


df = pd.read_csv("Data/data.csv", header=0)

# Data Wrangling

df.drop("id", axis=1, inplace=True)
df.drop("Unnamed: 32", axis=1, inplace=True)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

Y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(
    df, Y, random_state=42, test_size=0.25, shuffle=True
)


def eval_metrics(actual, pred):
    rmse = np.sqrt(metrics.mean_squared_error(actual, pred))
    mae = metrics.mean_absolute_error(actual, pred)
    r2 = metrics.r2_score(actual, pred)
    return rmse, mae, r2


n_estimators = float(sys.argv[1]) if len(sys.argv) > 1 else 100
min_samples_split = float(sys.argv[2]) if len(sys.argv) > 1 else 25
max_depth = float(sys.argv[3]) if len(sys.argv) > 1 else 7
max_features = float(sys.argv[4]) if len(sys.argv) > 1 else 2

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_depth=max_depth,
        max_features=max_features,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    (rmse, mae, r2) = eval_metrics(y_test, y_pred)

    print("Random Forest Classifier model (n_estimators={:f}, min_samples_split={:f},max_depth={:f},max_features={:f}):".format(n_estimators, min_samples_split,max_depth,max_features))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)


    # remote_server_uri = "https://dagshub.com/krishnaik06/mlflowexperiments.mlflow"
    # mlflow.set_tracking_uri(remote_server_uri)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(
            model, "model", registered_model_name="RandomForestBreastCancerClassifier"
        )
    else:
        mlflow.sklearn.log_model(model, "model")
