from lendingclub_scoring.common import Job
from lendingclub_scoring.config.ConfigProvider import read_config, setupMlflowConf
from lendingclub_scoring.pipelines.LendingClubABTestPipeline import LendingClubABTestPipeline


class ABTestJob(Job):

    def init_adapter(self):
        pass

    def launch(self):
        self.logger.info("Launching bootstrap job")

        setupMlflowConf(self.conf)
        p = LendingClubABTestPipeline(self.spark,
                                      self.conf['data-path'], self.conf['output-path'],
                                      self.conf['model-name'],
                                      self.conf['prod_version'], self.conf['test_version'])
        p.run()

        self.spark.read.load(self.conf['output-path']).show(10, False)

        self.logger.info("Bootstrap job finished!")


if __name__ == "__main__":
    job = ABTestJob()
    job.launch()
