# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import json
from lendingclub_scoring.common import Job
from lendingclub_scoring.config.ConfigProvider import setup_mlflow_config
from lendingclub_scoring.pipelines.LendingClubTrainingPipeline import (
    LendingClubTrainingPipeline,
)

# COMMAND ----------

conf = json.load(open("../conf/lendingclub_config.json"))
experiment_id = setup_mlflow_config(conf)
p = LendingClubTrainingPipeline(spark, conf["data-path"], conf["model-name"])
p.run()

# COMMAND ----------
