import unittest
from lendingclub_scoring.jobs.train_entrypoint import TrainJob
from uuid import uuid4
from pyspark.dbutils import DBUtils


class SampleJobIntegrationTest(unittest.TestCase):
    def setUp(self):

        self.test_dir = "dbfs:/tmp/tests/sample/%s" % str(uuid4())
        self.test_config = {"output_format": "delta", "output_path": self.test_dir}

        self.job = TrainJob(init_conf=self.test_config)
        self.dbutils = DBUtils(self.job.spark)
        self.spark = self.job.spark

    def test_sample(self):

        output_count = self.spark.range(1999).count()

        self.assertGreater(output_count, 0)

    def tearDown(self):
        self.dbutils.fs.rm(self.test_dir, True)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(SampleJobIntegrationTest("test_sample"))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
