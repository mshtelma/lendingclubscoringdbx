import json
from typing import Dict
import sys
from lendingclub_scoring.common import Job
from lendingclub_scoring.config.ConfigProvider import setup_mlflow_config
from lendingclub_scoring.pipelines.LendingClubModelEvaluationPipeline import (
    LendingClubModelEvaluationPipeline,
)


class WHModelEvalJob(Job):
    def __init__(self, webhook_conf: Dict[str, str]):
        self.webhook_conf = webhook_conf
        super().__init__()

    def init_adapter(self):
        self.conf = {
            "model_eval_mode": "promoted_candidates",
            "experiment-path": "/Shared/dbx/projects/lendingclubscoringdbx",
            "data-path": "dbfs:/databricks-datasets/samples/lending_club/parquet",
            "model-name": "LendingClubScoringModel",
            "training_webhook_model_eval_stage": "Staging",
            "model_eval_promote_model_to_stage": "Production",
            "model_deployment_type": "sagemaker",
            "sagemaker_endpoint_name": "shared-endpoint",
            "sagemaker_image_url": "997819012307.dkr.ecr.us-west-2.amazonaws.com/mlflow-pyfunc:1.11.0",
            "sagemaker_region": "us-west-2",
            "promoted_candidate_model_name": self.webhook_conf["model_name"],
            "promoted_candidate_model_version": self.webhook_conf["version"],
        }

    def launch(self):
        self.logger.info("Launching bootstrap job")
        to_stage = self.webhook_conf.get("to_stage")
        listening_stage = self.conf.get("training_webhook_model_eval_stage", "")
        if to_stage != listening_stage:
            print(
                f"The model was transferred to the stage {to_stage}, which is not configured stage {listening_stage}"
            )
            return

        experiment_id = setup_mlflow_config(self.conf)
        p = LendingClubModelEvaluationPipeline(self.spark, experiment_id, self.conf)
        p.run()

        self.logger.info("Bootstrap job finished!")


if __name__ == "__main__":
    print("Webhook request: ", sys.argv[1])
    _conf = json.loads(sys.argv[1])
    job = WHModelEvalJob(_conf)
    job.launch()
