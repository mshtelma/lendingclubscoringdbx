from lendingclub_scoring.common import JobContext
from lendingclub_scoring.config.ConfigProvider import setup_mlflow_config
from lendingclub_scoring.pipelines.LendingClubConsumerPipeline import (
    LendingClubConsumerPipeline,
)
from lendingclub_scoring.pipelines.LendingClubModelEvaluationPipeline import (
    LendingClubModelEvaluationPipeline,
)
from lendingclub_scoring.pipelines.LendingClubTrainingPipeline import (
    LendingClubTrainingPipeline,
)


def train():
    import sys

    print(sys.argv)
    ctx = JobContext()
    ctx.logger.info("Launching bootstrap job")
    _ = setup_mlflow_config(ctx.conf)
    p = LendingClubTrainingPipeline(ctx.spark, ctx.conf)
    p.run()
    ctx.logger.info("Bootstrap job finished!")


def eval():
    ctx = JobContext()
    ctx.logger.info("Launching bootstrap job")
    experiment_id = setup_mlflow_config(ctx.conf)
    p = LendingClubModelEvaluationPipeline(ctx.spark, experiment_id, ctx.conf)
    p.run()
    ctx.logger.info("Bootstrap job finished!")


def consume():
    ctx = JobContext()
    ctx.logger.info("Launching bootstrap job")
    _ = setup_mlflow_config(ctx.conf)
    p = LendingClubConsumerPipeline(
        ctx.spark,
        ctx.conf["data-path"],
        ctx.conf["output-path"],
        ctx.conf["model-name"],
        ctx.conf["stage"],
    )
    p.run()
    ctx.spark.read.load(ctx.conf["output-path"]).show(10, False)
    ctx.logger.info("Bootstrap job finished!")
