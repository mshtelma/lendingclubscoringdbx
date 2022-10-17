import json
from abc import ABC
from argparse import ArgumentParser
from logging import Logger
from typing import Dict, Any

from pyspark.sql import SparkSession


class JobContext(ABC):
    def __init__(self):
        self.spark = self._prepare_spark()
        self.logger = self._prepare_logger()
        self.conf = self._provide_config()
        self._log_conf()

    @staticmethod
    def _prepare_spark() -> SparkSession:
        return SparkSession.builder.getOrCreate()

    def _provide_config(self):
        self.logger.info("Reading configuration from --conf-file job option")
        conf_file = self._get_conf_file()
        if not conf_file:
            self.logger.info(
                "No conf file was provided, setting configuration to empty dict."
                "Please override configuration in subclass init method"
            )
            return {}
        else:
            self.logger.info(
                f"Conf file was provided, reading configuration from {conf_file}"
            )
            return self._read_config(conf_file)

    @staticmethod
    def _get_conf_file():
        p = ArgumentParser()
        p.add_argument("--conf-file", required=False, type=str)
        namespace = p.parse_known_args()[0]
        return namespace.conf_file

    def _read_config(self, conf_file) -> Dict[str, Any]:
        raw_content = "".join(
            self.spark.read.format("text").load(conf_file).toPandas()["value"].tolist()
        )
        config = json.loads(raw_content)
        return config

    def _prepare_logger(self) -> Logger:
        log4j_logger = self.spark._jvm.org.apache.log4j
        return log4j_logger.LogManager.getLogger(self.__class__.__name__)

    def _log_conf(self):
        # log parameters
        self.logger.info("Launching job with configuration parameters:")
        for key, item in self.conf.items():
            self.logger.info(f"\t Parameter: {key:<30} with value => {item:<30}")
