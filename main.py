import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import matplotlib.pyplot as plt  # side-stepping mpl backend
import matplotlib.gridspec as gridspec  # subplots

# Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import mlflow
import mlflow.sklearn


df = pd.read_csv("Data/diabetes.csv")

# Data Wrangling
diabets_df3 = df.copy()
zero_col = ["Glucose", "Insulin", "SkinThickness", "BloodPressure", "BMI"]
diabets_df3[zero_col] = diabets_df3[zero_col].replace(0, np.nan)
for col in ["Glucose", "Insulin", "SkinThickness"]:
    median_col = np.median(diabets_df3[diabets_df3[col].notna()][col])
    diabets_df3[col] = diabets_df3[col].fillna(median_col)
for col in ["BMI", "BloodPressure"]:
    mean_col = np.mean(diabets_df3[diabets_df3[col].notna()][col])
    diabets_df3[col] = diabets_df3[col].fillna(mean_col)


X = pd.DataFrame(
    diabets_df3,
    columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ],
)
Y = diabets_df3.Outcome
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=0, test_size=0.25
)

_max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 100
_penalty = (
    str(sys.argv[2]) if len(sys.argv) > 2 else "l2"
)  # ‘l1’, ‘l2’, ‘elasticnet’, None
_solver = (
    str(sys.argv[3]) if len(sys.argv) > 3 else "liblinear"
)  # ‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’

with mlflow.start_run():
    Logreg = LogisticRegression(penalty=_penalty, solver=_solver, max_iter=_max_iter)

    Logreg.fit(X_train, y_train)

    y_pred = Logreg.predict(X_test)
    y_predprob = Logreg.predict_proba(X_test)

    acurracy = metrics.accuracy_score(y_test, y_pred)
    precision_score = metrics.precision_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)

    print(
        "Logestic regression Classifier (penalty={}, solver={}, max_iter={}):".format(
            _penalty, _solver, _max_iter
        )
    )
    print("  acurracy: %s" % acurracy)
    print("  precision_score: %s" % precision_score)
    print("  recall_score: %s" % recall_score)
    print("  f1_score: %s" % f1_score)

    mlflow.log_param("penalty", _penalty)
    mlflow.log_param("solver", _solver)
    mlflow.log_param("max_iter", _max_iter)

    mlflow.log_metric("acurracy", acurracy)
    mlflow.log_metric("precision_score", precision_score)
    mlflow.log_metric("recall_score", recall_score)
    mlflow.log_metric("f1_score", f1_score)

    remote_server_uri = "https://dagshub.com/J3mai/MlOps.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.sklearn.log_model(Logreg, "Logreg")
    mlflow.end_run()
