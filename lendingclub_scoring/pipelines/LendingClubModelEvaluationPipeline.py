from typing import Dict

import mlflow
import mlflow.sklearn
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.metrics import roc_auc_score

from lendingclub_scoring.data.DataProvider import LendingClubDataProvider
from lendingclub_scoring.deploy import deploy_to_sagemaker


class LendingClubModelEvaluationPipeline:
    def __init__(
        self,
        spark: SparkSession,
        experiment_id: str,
        conf: Dict[str, str],
        limit: int = None,
    ):
        self.spark = spark
        self.conf = conf
        self.input_path = self.conf["data-path"]
        self.model_name = self.conf["model-name"]
        self.limit = limit
        self.experiment_id = experiment_id
        self.data_provider = LendingClubDataProvider(spark, self.input_path, limit)

    def run(self):
        mlflow_client = MlflowClient()

        _, x_test, _, y_test = self.data_provider.run()
        cand_run_ids = self.get_candidate_models()
        best_cand_roc, best_cand_run_id = self.get_best_model(
            cand_run_ids, x_test, y_test
        )
        print("Best ROC (candidate models): ", best_cand_roc)

        try:
            versions = mlflow_client.get_latest_versions(
                self.model_name, stages=["Production"]
            )
            prod_run_ids = [v.run_id for v in versions]
            best_prod_roc, best_prod_run_id = self.get_best_model(
                prod_run_ids, x_test, y_test
            )
        except RestException:
            best_prod_roc = -1
        print("ROC (production models): ", best_prod_roc)

        if best_cand_roc >= best_prod_roc:
            # deploy new model
            model_uri = f"runs:/{best_cand_run_id}/model"
            model_version = mlflow.register_model(model_uri, self.model_name)
            # time.sleep(15)
            mlflow_client.transition_model_version_stage(
                name=self.model_name, version=model_version.version, stage="Production"
            )
            prod_metric = best_cand_roc
            prod_run_id = best_cand_run_id
            deployed = True
            print("Deployed version: ", model_version.version)
            if self.conf.get("model_deployment_type", "none") == "sagemaker":
                deploy_to_sagemaker(
                    self.conf["sagemaker_endpoint_name"],
                    self.conf["sagemaker_image_url"],
                    model_uri,
                    self.conf["sagemaker_region"],
                )
        else:
            prod_metric = best_prod_roc
            prod_run_id = best_prod_run_id
            deployed = False

        with mlflow.start_run(run_name="Evaluation"):
            mlflow.log_metric("prod_metric", best_prod_roc)
            mlflow.log_metric("cand_metric", best_cand_roc)
            mlflow.log_metric("prod_metric", prod_metric)
            mlflow.log_param("prod_run_uuid", prod_run_id)
            mlflow.log_param("deployed", deployed)
            mlflow.set_tag("action", "model-eval")

        # remove candidate tags
        # for run_id in cand_run_ids:
        #    mlflow_client.set_tag(run_id, 'candidate', 'false')

    def get_best_model(self, run_ids, x, y):
        best_roc = -1
        best_run_id = None
        for run_id in run_ids:
            roc = self.evaluate_model(run_id, x, y)
            if roc > best_roc:
                best_roc = roc
                best_run_id = run_id
        return best_roc, best_run_id

    def get_candidate_models(self):
        spark_df = self.spark.read.format("mlflow-experiment").load(self.experiment_id)
        pdf = spark_df.where("tags.candidate='true'").select("run_id").toPandas()
        return pdf["run_id"].values

    def evaluate_model(self, run_id, x, y):
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        predictions = model.predict(x)
        # acc = accuracy_score(Y, predictions)
        roc = roc_auc_score(y, predictions)
        return roc
