import time
import mlflow
import mlflow.sklearn
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, roc_auc_score

from lendingclub_scoring.data.DataProvider import LendingClubDataProvider


class LendingClubModelEvaluationPipeline():
    def __init__(self, spark, experimentID, model_name, input_path, limit=None):
        self.spark = spark
        self.input_path = input_path
        self.model_name = model_name
        self.limit = limit
        self.experimentID = experimentID
        self.data_provider = LendingClubDataProvider(spark, input_path, limit)

    def run(self):
        mlflow_client = MlflowClient()

        _, X_test, _, Y_test = self.data_provider.run()
        cand_run_ids = self.get_candidate_models()
        best_cand_roc, best_cand_run_id = self.get_best_model(cand_run_ids, X_test, Y_test)
        print('Best ROC (candidate models): ', best_cand_roc)

        try:
            versions = mlflow_client.get_latest_versions(self.model_name, stages=['Production'])
            prod_run_ids = [v.run_id for v in versions]
            best_prod_roc, best_prod_run_id = self.get_best_model(prod_run_ids, X_test, Y_test)
        except RestException:
            best_prod_roc = -1
        print('ROC (production models): ', best_prod_roc)

        if best_cand_roc >= best_prod_roc:
            # deploy new model
            model_version = mlflow.register_model("runs:/" + best_cand_run_id + "/model", self.model_name)
            time.sleep(15)
            mlflow_client.transition_model_version_stage(name=self.model_name, version=model_version.version,
                                                         stage="Production")
            prod_metric = best_cand_roc
            prod_run_id = best_cand_run_id
            deployed = True
            print('Deployed version: ', model_version.version)
        else:
            prod_metric = best_prod_roc
            prod_run_id = best_prod_run_id
            deployed = False

        with mlflow.start_run(run_name="Evaluation") as run:
            mlflow.log_metric("prod_metric", best_prod_roc)
            mlflow.log_metric("cand_metric", best_cand_roc)
            mlflow.log_metric("prod_metric", prod_metric)
            mlflow.log_param("prod_run_uuid", prod_run_id)
            mlflow.log_param("deployed", deployed)
            mlflow.set_tag("action", "model-eval")

        # remove candidate tags
        for run_id in cand_run_ids:
            mlflow_client.set_tag(run_id, 'candidate', 'false')

    def get_best_model(self, run_ids, X, Y):
        best_roc = -1
        best_run_id = None
        for run_id in run_ids:
            roc = self.evaluate_model(run_id, X, Y)
            if roc > best_roc:
                best_roc = roc
                best_run_id = run_id
        return best_roc, best_run_id

    def get_candidate_models(self):
        spark_df = self.spark.read.format("mlflow-experiment").load(self.experimentID)
        pdf = spark_df.where("tags.candidate='true'").select("run_id").toPandas()
        return pdf['run_id'].values

    def evaluate_model(self, run_id, X, Y):
        model = mlflow.sklearn.load_model('runs:/{}/model'.format(run_id))
        predictions = model.predict(X)
        # acc = accuracy_score(Y, predictions)
        roc = roc_auc_score(Y, predictions)
        return roc
