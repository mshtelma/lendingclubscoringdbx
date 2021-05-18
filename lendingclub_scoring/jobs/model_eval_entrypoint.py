from lendingclub_scoring.common import Job
from lendingclub_scoring.config.ConfigProvider import read_config, setupMlflowConf
from lendingclub_scoring.pipelines.LendingClubModelEvaluationPipeline import LendingClubModelEvaluationPipeline


class ModelEvalJob(Job):

    def init_adapter(self):
        if not self.conf:
            self.logger.info("Init configuration was not provided, using configuration from default_init method")
            self.conf = {
                "experiment-path": "/Users/michael.shtelma@databricks.com/experiments/leclub1",
                "data-path": "dbfs:/databricks-datasets/samples/lending_club/parquet",
                "model-name": "LendingClubScoringModel"
            }
        else:
            self.logger.info("Init configuration is already provided")

    def launch(self):
        self.logger.info("Launching bootstrap job")

        experimentID = setupMlflowConf(self.conf)
        p = LendingClubModelEvaluationPipeline(self.spark, experimentID, self.conf['model-name'],
                                               self.conf['data-path'])
        p.run()

        self.logger.info("Bootstrap job finished!")


if __name__ == "__main__":
    job = ModelEvalJob()
    job.launch()
