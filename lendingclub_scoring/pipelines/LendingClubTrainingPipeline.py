from typing import Dict

import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from lendingclub_scoring.data.DataProvider import LendingClubDataProvider
from lendingclub_scoring.webhooks import setup_webhook_for_model


class LendingClubTrainingPipeline:
    def __init__(self, spark: SparkSession, conf: Dict[str, str], limit=None):
        self.spark = spark
        self.conf = conf
        self.input_path = self.conf["data-path"]
        self.model_name = self.conf["model-name"]
        self.limit = limit
        self.data_provider = LendingClubDataProvider(spark, self.input_path, limit)

    def run(self):
        x_train, x_test, y_train, y_test = self.data_provider.run()
        self.train(x_train, x_test, y_train, y_test)

    def train(self, x_train, x_test, y_train, y_test):
        cl = LogisticRegression(random_state=42, max_iter=10)
        # cl = RandomForestClassifier(n_estimators=20)
        cl.fit(x_train, y_train)
        with mlflow.start_run(run_name="Training"):
            self.eval_and_log_metrics(cl, x_test, y_test)
            _model_name = None
            if self.conf.get("training_promote_candidates", False):
                _model_name = self.model_name
            mlflow.sklearn.log_model(cl, "model", registered_model_name=_model_name)
            mlflow.set_tag("action", "training")
            if self.conf.get("training_webhook_for_model_eval", False):
                setup_webhook_for_model(
                    self.model_name,
                    self.conf["training_model_eval_job_id"],
                    self.conf.get(
                        "training_webhook_event", "MODEL_VERSION_TRANSITIONED_STAGE"
                    ),
                )

    def eval_and_log_metrics(self, estimator, x, y):
        predictions = estimator.predict(x)

        # Calc metrics
        acc = accuracy_score(y, predictions)
        roc = roc_auc_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        # Print metrics
        print(f"  acc: {acc}")
        print(f"  roc: {roc}")
        print(f"  mse: {mse}")
        print(f"  mae: {mae}")
        print(f"  R2: {r2}")

        # Log metrics
        mlflow.log_metric("acc", acc)
        mlflow.log_metric("roc", roc)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.set_tag("candidate", "true")
