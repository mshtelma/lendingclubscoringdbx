import unittest
import tempfile
from pyspark.sql import SparkSession


class SampleJobUnitTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory().name
        self.spark = SparkSession.builder.master("local[1]").getOrCreate()

    def test_sample(self):

        output_count = self.spark.range(100).count()

        self.assertGreater(output_count, 0)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
