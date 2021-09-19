import mlflow
import mlflow.sklearn
from pyspark.sql import Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from lendingclub_scoring.data.DataProvider import LendingClubDataProvider


class LendingClubTrainingPipeline():
    def __init__(self, spark, input_path, model_name, limit=None):
        self.spark = spark
        self.input_path = input_path
        self.model_name = model_name
        self.limit = limit
        self.data_provider = LendingClubDataProvider(spark, input_path, limit)

    def run(self):
        X_train, X_test, Y_train, Y_test = self.data_provider.run()
        self.train(X_train, X_test, Y_train, Y_test)

    def train(self, X_train, X_test, Y_train, Y_test):
        #cl = LogisticRegression(random_state=42, max_iter=15)
        cl = RandomForestClassifier(n_estimators=20)
        cl.fit(X_train, Y_train)
        with mlflow.start_run(run_name="Training") as run:
            self.eval_and_log_metrics(cl, X_test, Y_test)
            mlflow.sklearn.log_model(cl, "model")
            mlflow.set_tag("action", "training")

    def eval_and_log_metrics(self, estimator, X, Y):
        predictions = estimator.predict(X)

        # Calc metrics
        acc = accuracy_score(Y, predictions)
        roc = roc_auc_score(Y, predictions)
        mse = mean_squared_error(Y, predictions)
        mae = mean_absolute_error(Y, predictions)
        r2 = r2_score(Y, predictions)

        # Print metrics
        print("  acc: {}".format(acc))
        print("  roc: {}".format(roc))
        print("  mse: {}".format(mse))
        print("  mae: {}".format(mae))
        print("  R2: {}".format(r2))

        # Log metrics
        mlflow.log_metric("acc", acc)
        mlflow.log_metric("roc", roc)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.set_tag('candidate', 'true')
