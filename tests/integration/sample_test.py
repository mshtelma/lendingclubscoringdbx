import unittest
from lendingclub_scoring.jobs.train_entrypoint import TrainJob
from uuid import uuid4
from pyspark.dbutils import DBUtils


class SampleJobIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = f"dbfs:/tmp/tests/sample/{str(uuid4())}"
        self.test_config = {
            "output_format": "delta",
            "output_path": self.test_dir,
            "model-name": "LendingClubScoringModel",
            "data-path": "dbfs:/databricks-datasets/samples/lending_club/parquet",
            "experiment-path": "/Shared/leclub_test",
        }

        self.job = TrainJob(init_conf=self.test_config)
        self.dbutils = DBUtils(self.job.spark)
        self.spark = self.job.spark

    def test_sample(self):
        self.job.launch()

        spark_df = (
            self.spark.read.format("mlflow-experiment")
            .load(self.job.experiment_id)
            .where("tags.candidate='true'")
        )

        self.assertGreater(spark_df.count(), 0)

    def tearDown(self):
        self.dbutils.fs.rm(self.test_dir, True)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(SampleJobIntegrationTest("test_sample"))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
