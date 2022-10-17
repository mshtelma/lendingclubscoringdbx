import json
import mlflow.tracking
from mlflow.utils.rest_utils import http_request


def mlflow_call_endpoint(endpoint, method, body, creds):
    if method == "GET":
        response = http_request(
            host_creds=creds,
            endpoint=f"/api/2.0/mlflow/{endpoint}",
            method=method,
            params=json.loads(body),
        )
    else:
        response = http_request(
            host_creds=creds,
            endpoint=f"/api/2.0/mlflow/{endpoint}",
            method=method,
            json=json.loads(body),
        )
    return response.json()


def setup_webhook_for_model(model_name: str, job_id: str, event: str):
    from databricks_registry_webhooks import RegistryWebhooksClient

    for wh in RegistryWebhooksClient().list_webhooks(model_name=model_name):
        RegistryWebhooksClient().delete_webhook(wh.id)
    client = mlflow.tracking.client.MlflowClient()
    host_creds = client._tracking_client.store.get_host_creds()
    trigger_job = json.dumps(
        {
            "model_name": model_name,
            "events": [event],
            "description": "Trigger the Model Evaluation Pipeline when a model is moved to Staging",
            "status": "ACTIVE",
            "job_spec": {
                "job_id": job_id,
                "workspace_url": host_creds.host,
                "access_token": host_creds.token,
                "notebook_params": {
                    "event_message": "<Webhook Payload>",
                },
            },
        }
    )

    mlflow_call_endpoint("registry-webhooks/create", "POST", trigger_job, host_creds)
