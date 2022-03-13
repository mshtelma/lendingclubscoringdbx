import mlflow
import mlflow.sagemaker as mfs


def deploy_to_sagemaker(
    endpoint_name: str, image_url: str, model_uri: str, region_name: str
):
    mfs.deploy(
        app_name=endpoint_name,
        model_uri=model_uri,
        region_name=region_name,
        mode=mlflow.sagemaker.DEPLOYMENT_MODE_ADD,  # "replace",
        image_url=image_url,
    )
