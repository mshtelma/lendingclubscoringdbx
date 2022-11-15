from uuid import uuid4

from lendingclub_scoring.common import JobContext
from lendingclub_scoring.config.ConfigProvider import setup_mlflow_config
from lendingclub_scoring.main_entrypoint import train


def test_train_int():
    ctx = JobContext()
    exp_id = setup_mlflow_config(ctx.conf)

    train()

    spark_df = (
        ctx.spark.read.format("mlflow-experiment")
        .load(exp_id)
        .where("tags.candidate='true'")
    )

    assert spark_df.count() > 0
