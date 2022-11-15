from typing import Dict

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.linear_model import LogisticRegression

from lendingclub_scoring.data.DataProvider import LendingClubDataProvider
from lendingclub_scoring.webhooks import setup_webhook_for_model


class LendingClubTrainingPipeline:
    def __init__(self, spark: SparkSession, conf: Dict[str, str], limit=100000):
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
        mlflow.sklearn.autolog()
        # cl = LogisticRegression(random_state=42, max_iter=10)
        with mlflow.start_run(run_name="Training") as run:
            cl = LogisticRegression(random_state=42, max_iter=10)
            cl.fit(x_train, y_train)
            signature = infer_signature(x_train, y_train)
            _model_name = None
            if self.conf.get("training_promote_candidates", False):
                _model_name = self.model_name
            mlflow.sklearn.log_model(
                cl, "model", registered_model_name=_model_name, signature=signature
            )
            mlflow.set_tag("action", "training")
            self.eval_and_log_metrics(
                f"runs:/{run.info.run_uuid}/model", x_test, y_test
            )
            if self.conf.get("training_webhook_for_model_eval", False):
                setup_webhook_for_model(
                    self.model_name,
                    self.conf["training_model_eval_job_id"],
                    self.conf.get(
                        "training_webhook_event", "MODEL_VERSION_TRANSITIONED_STAGE"
                    ),
                )

    def eval_and_log_metrics(self, model_uri, x, y):
        _df = x.copy()
        _df["y"] = y
        mlflow.evaluate(
            model=model_uri,
            data=_df,
            targets="y",
            model_type="classifier",
            evaluator_config={"log_model_explainability": False},
        )
        mlflow.set_tag("candidate", "true")
