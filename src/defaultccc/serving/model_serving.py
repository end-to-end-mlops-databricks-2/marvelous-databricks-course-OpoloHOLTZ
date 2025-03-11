import os

import mlflow
import numpy as np
import pandas as pd
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


class ModelServing:
    def __init__(self, endpoint_name: str, model_name: str):
        """
        Initialize the model serving manager.
        """
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def get_latest_model_version(self):
        """
        Get the lastest version of the model.
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        logger.info(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self, version: str = "latest", workload_size: str = "Small", scale_to_zero: bool = True
    ):
        """
        Deploys the model serving endpoint in Databricks.

        :param version: str. The version of the model to deploy.
        :param workload_seze: str. Workload size. The default is "Small" which is 4 concurrent requests.
        :param scale_to_zero: bool. If True, the endpoint scales to 0 when unused.
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        if version == "latest":
            entity_version = self.get_latest_model_version()
        else:
            entity_version = version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)

    def call_endpoint(self, record: list[dict]):
        """
        Calls the model serving endpoint with a given input record.
        """
        serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/default_ccc-model-serving/invocations"
        response = requests.post(
            serving_endpoint,
            headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
            json={"dataframe_records": record},
        )
        return response.status_code, response.json() if response.status_code == 200 else response.text

    def evaluate_serving_model(self, input_data: pd.DataFrame, target: str):
        """
        Evaluate the deployed model by sending predictions to the serving endpoint.
        """
        logger.info("Evaluating the deployed model via the serving endpoint...")
        records = input_data.to_dict(orient="records")
        status_code, response = self.call_endpoint(records)

        if status_code == 200:
            y_pred = np.array(response["predictions"]).flatten()
            y_true = input_data[target].values

            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "roc_auc": roc_auc_score(y_true, y_pred),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            }

            logger.info(f"Serving Model Evaluation: {metrics}")
            return metrics
        else:
            logger.error(f"Error calling serving endpoint: {response}")
            return None
