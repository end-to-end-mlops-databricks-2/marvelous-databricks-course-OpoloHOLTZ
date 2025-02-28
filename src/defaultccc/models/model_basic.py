import mlflow
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient

from defaultccc.config import ProjectConfig, Tags

class BasicModel():
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        self.config = config
        self.spark = spark

        # From config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.parameters = self.config.parameters
        self.experiment_name = self.config.experiment_name_basic
        self.data_version = "0"
        self.tags = tags.dict()
        self.model_name = f"{self.catalog_name}.{self.schema_name}.default_ccc_model_basic"
    
    def load_data(self):
        """
        Load the split the data into X and y for train and test.
        """

        logger.info(f"Loading data from {self.config.catalog_name}.{self.config.schema_name} tables train and test...")
        
        self.train_set = self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.train_set")
        self.train_set_pd = self.train_set.toPandas()
        self.test_set = self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.test_set")
        self.test_set_pd = self.test_set.toPandas()

        # Separate features
        self.X_train = self.train_set_pd[self.num_features + self.cat_features]
        self.y_train = self.train_set_pd[self.target]
        self.X_test = self.test_set_pd[self.num_features + self.cat_features]
        self.y_test = self.test_set_pd[self.target]

        logger.info(f"Data succesfully loaded.")

    def prepare_features(self):
        """
        Use a pipeline to preprocess features.
        Transformer:
            categorial features ->  OneHotEncoder
            numerical features -> StandardScaler
        Model:
            -Logit
        """
        logger.info("Starting the preprocesing with a pipeline...")

        self.preprocessor = ColumnTransformer(
            transformers = [("categorial", OneHotEncoder(handle_unknown="ignore"), self.cat_features),
                           ("numerical", StandardScaler(), self.num_features)
                           ],
                             remainder="passthrough"
        )
        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), 
                   ("classificator", LogisticRegression(**self.parameters))]
        )

        logger.info("Preprocessing data pipeline succeded")

    def train_model(self):
        """
        Train the model using the pipeline
        """
        logger.info("Strating to train...")
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("Model training completed.")

    def evaluate_model(self):
        """
        Evaluate the model and log metrics.
        """
        logger.info("Evaluating the model...")
        
        y_pred = self.pipeline.predict(self.X_test)
        y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]
        
        # MLflow metrics logger
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        cm = confusion_matrix(self.y_test, y_pred)
        
        logger.info(f"Model Evaluation:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nROC AUC: {roc_auc}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "y_pred": y_pred
        }

    def log_model(self):
        """
        Logs for the model.
        """
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            metrics = self.evaluate_model()
            
            for metric_name, metric_value in metrics.items():
                if metric_name != "confusion_matrix":
                    mlflow.log_param("model_type", "Logistic Regression Classificator")
                    mlflow.log_params(self.parameters)
                    if isinstance(metric_value, (list, np.ndarray)):
                        if len(metric_value) == 1:
                            metric_value = float(metric_value[0])
                        else:
                            continue
                    mlflow.log_metric(metric_name, metric_value)

            signature = infer_signature(model_input=self.X_train, model_output=metrics["y_pred"])
            dataset = mlflow.data.from_spark(
                self.train_set,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version
            )
            mlflow.log_input(
                dataset,
                context="training"
            )
            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="logit_pipeline_model", 
                signature=signature
            )

            logger.info(f"Model logged successfully with Run ID: {self.run_id}")

    def register_model(self):
        """
        Register the model in the unity catalog.
        """
        logger.info("Registering the model in the UC...")

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/logit_pipeline_model",
            name=f"{self.catalog_name}.{self.schema_name}.default_ccc_model_basic",
            tags=self.tags
        )

        logger.info(f"Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.default_ccc_model_basic",
            alias="latest-model", # the alias name is mandatory for databricks
            version=latest_version
        )

    def retrieve_current_run_dataset(self):
        """
        Retrieve the run dataset.
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)

        logger.info("Dataset loaded.")

        return dataset_source.load()

    def retrieve_current_run_metadata(self):
        """
        Retrieve the run metadata.
         
        :return metrics: metrics saved from the experience/model linked to the run_id.
        :return parameters: parameters used from the experience/model linked to the run_id.
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]

        logger.info("Dataset metadata loaded.")

        return metrics, params
    
    def load_latest_model_and_predict(self, input_data: pd.DataFrame):
        """
        Load the lastest, alias="latest-model" saved from register_model.
        Predict the input data with the latest model.

        :param input_data: Pandas dataframe.
        
        :return predictions: Input data predicted.
        """
        # Load latest version
        logger.info("Load the model from MLFlow.")

        model_uri = f"models:/{self.model_name}@latest-model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("Model is successfully loaded.")

        # Predict
        predictions = model.predict(input_data)

        return predictions

# Not working yet
    # @staticmethod
    # def delete_models_by_experiment(experiment_name: str, filter_string: str, model_name: str):
    #     """
    #     Deletes all versions of a model associated with the runs of a given MLflow experiment.

    #     :param experiment_name: Name of the experiment in MLflow.
    #     :param filter_string: Filter to select a run.
    #     :param model_name: Name of the model to delete.
    #     """
    #     client = MlflowClient()

    #     # Look for the runs filtered
    #     runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter_string)

    #     if runs.empty:
    #         logger.warning(f"No run found for {experiment_name} with the filter '{filter_string}'")
    #         return

    #     logger.info(f"Found {len(runs)} runs for {experiment_name} with the filter '{filter_string}'")

    #     # List the run_id
    #     run_ids = runs["run_id"].tolist()

    #     # Delete the version linked to his run_id
    #     for run_id in run_ids:
    #         versions = client.search_model_versions(f"run_id='{run_id}'")

    #         if not versions:
    #             logger.warning(f"No version found for the run_id {run_id}")
    #             continue

    #         for v in versions:
    #             if v.name == model_name:
    #                 client.delete_model_version(name=model_name, version=v.version)
    #                 logger.info(f"Deleted : {model_name} version {v.version} for the run {run_id}")
