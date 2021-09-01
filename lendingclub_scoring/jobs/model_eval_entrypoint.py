from lendingclub_scoring.common import Job
from lendingclub_scoring.config.ConfigProvider import read_config, setupMlflowConf
from lendingclub_scoring.pipelines.LendingClubModelEvaluationPipeline import LendingClubModelEvaluationPipeline


class ModelEvalJob(Job):

    def init_adapter(self):
        pass

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
