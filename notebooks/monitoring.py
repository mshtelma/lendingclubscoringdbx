# Databricks notebook source
# MAGIC %sql
# MAGIC create database if not exists msh;

# COMMAND ----------

import mlflow
experiment_path = "/Users/Michael.Shtelma@databricks.com/experiments/leclub1"
mlflow.set_experiment(experiment_path)
experimentID = mlflow.get_experiment_by_name(experiment_path).experiment_id
mlflow_df = spark.read.format("mlflow-experiment").load(experimentID).where("tags.action='model-eval'").select("start_time", "params.deployed", "params.prod_run_uuid", "metrics.prod_metric", "metrics.cand_metric")
display(mlflow_df)

# COMMAND ----------

mlflow_df.write.mode("overwrite").saveAsTable("msh.lending_club_model_eval_monitoring")

# COMMAND ----------

mlflow_df = spark.read.format("mlflow-experiment").load(experimentID).where("tags.action<>'model-eval'").select("start_time", "metrics.roc", "metrics.acc", "metrics.mae")
display(mlflow_df)

# COMMAND ----------

mlflow_df.write.mode("overwrite").saveAsTable("msh.lending_club_training_metrics")
