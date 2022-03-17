from typing import Dict, List, Tuple

import mlflow
import mlflow.sklearn
from mlflow.entities.model_registry import ModelVersion
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
        cand_model_version = None

        _, x_test, _, y_test = self.data_provider.run()
        if self.conf.get("model_eval_mode", "simple_runs") == "promoted_candidates":
            print(
                f'Searhing for Promoted candidates for {self.conf["promoted_candidate_model_name"]} model '
                + f'and version {self.conf["promoted_candidate_model_version"]}...'
            )
            cand_run_id, cand_model_version = self.get_run_id_for_model_version(
                self.conf["promoted_candidate_model_name"],
                self.conf["promoted_candidate_model_version"],
            )
            cand_run_ids = [cand_run_id]
        else:
            print("Searhing for Simple Candidate runs...")
            cand_run_ids = self.get_candidate_models()
        print(f"Found following candidate runs: {cand_run_ids}")
        best_cand_roc, best_cand_run_id = self.get_best_model(
            cand_run_ids, x_test, y_test
        )
        print("Best ROC (candidate models): ", best_cand_roc)

        try:
            prod_run_ids = self.get_promoted_candidate_models(stage="Production")
            best_prod_roc, best_prod_run_id = self.get_best_model(
                prod_run_ids, x_test, y_test
            )
        except RestException:
            best_prod_roc = -1
        print("ROC (production models): ", best_prod_roc)

        if best_cand_roc >= best_prod_roc:
            print("Deploying new model...")
            # deploy new model
            model_uri = f"runs:/{best_cand_run_id}/model"
            if cand_model_version is None:
                model_version = mlflow.register_model(model_uri, self.model_name)
            else:
                model_version = cand_model_version
            mlflow_client.transition_model_version_stage(
                name=self.model_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True,
            )
            prod_metric = best_cand_roc
            prod_run_id = best_cand_run_id
            deployed = True
            print("Deployed version: ", model_version.version)
            if self.conf.get("model_deployment_type", "none") == "sagemaker":
                print("Performing SageMaker deployment...")
                try:
                    deploy_to_sagemaker(
                        self.conf["sagemaker_endpoint_name"],
                        self.conf["sagemaker_image_url"],
                        model_uri,
                        self.conf["sagemaker_region"],
                    )
                except Exception as e:
                    print(
                        "Error has occured while deploying the model to SageMaker: ", e
                    )
        else:
            print(
                "Candidate models are not better that the one we have currently in Production. Doing nothing..."
            )
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

    def get_candidate_models(self) -> List[str]:
        spark_df = self.spark.read.format("mlflow-experiment").load(self.experiment_id)
        pdf = spark_df.where("tags.candidate='true'").select("run_id").toPandas()
        return pdf["run_id"].values

    def get_promoted_candidate_models(self, stage: str = "None") -> List[str]:
        mlflow_client = MlflowClient()
        versions = mlflow_client.get_latest_versions(self.model_name, stages=[stage])
        return [v.run_id for v in versions]

    def get_run_id_for_model_version(
        self, model_name: str, model_version: str
    ) -> Tuple[str, ModelVersion]:
        mlflow_client = MlflowClient()
        mlflow_model_version = mlflow_client.get_model_version(
            model_name, model_version
        )
        return mlflow_model_version.run_id, mlflow_model_version

    def evaluate_model(self, run_id, x, y):
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        predictions = model.predict(x)
        # acc = accuracy_score(Y, predictions)
        roc = roc_auc_score(y, predictions)
        return roc
