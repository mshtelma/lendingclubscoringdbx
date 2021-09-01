
from lendingclub_scoring.common import Job
from lendingclub_scoring.config.ConfigProvider import read_config, setupMlflowConf
from lendingclub_scoring.pipelines.LendingClubTrainingPipeline import LendingClubTrainingPipeline




class TrainJob(Job):

    def init_adapter(self):
        pass

    def launch(self):
        self.logger.info("Launching bootstrap job")

        setupMlflowConf(self.conf)
        p = LendingClubTrainingPipeline(self.spark, self.conf['data-path'], self.conf['model-name'])
        p.run()

        self.logger.info("Bootstrap job finished!")


if __name__ == "__main__":
    job = TrainJob()
    job.launch()
