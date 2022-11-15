import mlflow
import pytest
from mlflow import MlflowClient
from pyspark.sql import SparkSession, DataFrame

from lendingclub_scoring.pipelines.LendingClubTrainingPipeline import (
    LendingClubTrainingPipeline,
)


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.master("local[1]").getOrCreate()


@pytest.fixture
def test_data_sdf(shared_datadir, spark) -> DataFrame:
    return spark.read.parquet(str(shared_datadir / "lendingclub.parquet"))


def test_train_pipeline(test_data_sdf, spark):
    mlflow_exp_path = "/tmp/lendingclubscoringdbx"
    mlflow.set_experiment(mlflow_exp_path)
    exp_id = mlflow.get_experiment_by_name(mlflow_exp_path).experiment_id
    test_data_sdf.createOrReplaceTempView("train")
    conf = {
        "data-path": "select * from train",
        "model-name": "LendingClubScoringModel",
    }
    p = LendingClubTrainingPipeline(spark, conf, limit=1000)
    p.run()
    runs = MlflowClient().search_runs(experiment_ids=[exp_id])
    assert len(runs) > 0
