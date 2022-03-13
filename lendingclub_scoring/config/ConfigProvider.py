import time
import mlflow
import yaml


def read_config(name, root):
    try:
        filename = root.replace("dbfs:", "/dbfs") + "/" + name
        with open(filename, encoding="utf-8") as conf_file:
            conf = yaml.load(conf_file, Loader=yaml.FullLoader)
            return conf
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}. Please include a config file!")


def setup_mlflow_config(conf):
    mlflow.set_experiment(conf["experiment-path"])
    try:
        experiment_id = mlflow.get_experiment_by_name(
            conf["experiment-path"]
        ).experiment_id
        return experiment_id
    except FileNotFoundError:
        time.sleep(10)
        experiment_id = mlflow.get_experiment_by_name(
            conf["experiment-path"]
        ).experiment_id
        return experiment_id
