import mlflow
import pandas as pd
from pyspark.sql import SparkSession
from loguru import logger

from defaultccc.config import ProjectConfig, Tags
from defaultccc.models.model_basic import BasicModel

mlflow.set_tracking_uri("databrics")
mlflow.set_registry_uri("databrics-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
tags = Tags(**tags_dict)

# Initialize the model
basic_model = BasicModel(config=config, tags=tags, spark=spark)
basic_model.load_data()
basic_model.prepare_features()

# Train and log the model
basic_model.train_model()
basic_model.log_model()

# Register model
basic_model.register_model()

# Search for an experiment
run_id = mlflow.search_runs(
    experiment_names=["/Shared/default-ccc-basic"],
    filter_string="tags.branch='week2'"
)["run_id"].iloc[0]

model = mlflow.sklearn.load_model(model_uri=f"runs:/{run_id}/logit_pipeline_model")

# Retrieve the dataset 
basic_model.retrieve_current_run_dataset()

# Retrievethe dataset metadata 
basic_model.retrieve_current_run_metadata()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

prediction_df = basic_model.load_latest_model_and_predict(test_set[test_set.columns[~test_set.columns.isin([config.target])]])

print(prediction_df)



# # Try to delete a model by experiment but not working yet.

# BasicModel.delete_models_by_experiment(
#     experiment_name="/Shared/default-ccc-basic",
#     filter_string="tags.branch='week2'",
#     model_name=f"{config.catalog_name}.{config.schema_name}.default_ccc_model_basic"
# )

# # Find the model name
# from mlflow import MlflowClient
# client = MlflowClient()
# models = client.search_registered_models()
# for m in models:
#     print(m.name)



    