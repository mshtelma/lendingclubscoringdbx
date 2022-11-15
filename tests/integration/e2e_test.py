from lendingclub_scoring.common import JobContext
from lendingclub_scoring.config.ConfigProvider import setup_mlflow_config
from lendingclub_scoring.main_entrypoint import train
from lendingclub_scoring.pipelines.LendingClubTrainingPipeline import (
    LendingClubTrainingPipeline,
)


def test_train_int():
    ctx = JobContext()
    conf = {
        "output_format": "delta",
        "experiment-path": "/Shared/dbx/projects/lendingclubscoringdbx_int_test",
        "data-path": "dbfs:/databricks-datasets/samples/lending_club/parquet",
        "output-path": "dbfs:/tmp/leout_int_test",
        "model-name": "LendingClubScoringModelIntTest",
        "training_promote_candidates": False,
        "training_webhook_for_model_eval": False,
        "training_webhook_model_eval_stage": "Staging",
        "model_eval_promote_model_to_stage": "Production",
        "model_deployment_type": "none",
        "consumer_stage": "Production",
    }
    exp_id = setup_mlflow_config(conf)

    p = LendingClubTrainingPipeline(ctx.spark, conf)
    p.run()

    spark_df = (
        ctx.spark.read.format("mlflow-experiment")
        .load(exp_id)
        .where("tags.candidate='true'")
    )

    assert spark_df.count() > 0
