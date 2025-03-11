import argparse

import mlflow
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

from defaultccc.config import ProjectConfig, Tags
from defaultccc.models.model_basic import BasicModel

mlflow.set_tracking_uri("databricks://opoloholtz")
mlflow.set_registry_uri("databricks-uc://opoloholtz")

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--root_path",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )

# parser.add_argument(
#     "--env",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )

# parser.add_argument(
#     "--git_sha",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )

# parser.add_argument(
#     "--job_run_id",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )

# parser.add_argument(
#     "--branch",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )

# args = parser.parse_args()
# root_path = args.root_path
# config_path = f"{root_path}/files/project_config.yml"

args = argparse.Namespace(
    root_path="/Workspace/Users/opolo.holtz@amaris.com/.bundle/marvelous-databricks-course-OpoloHOLTZ/dev",
    env="dev",
    git_sha="abcd12345",
    job_run_id="0",
    branch="week4"
)
config_path = "../project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# config = ProjectConfig.from_yaml(config_path="../project_config.yml")
# spark = SparkSession.builder.getOrCreate()
# tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
# tags = Tags(**tags_dict)

# Initialize the model
basic_model = BasicModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")
basic_model.load_data()
logger.info("Data loaded.")
basic_model.prepare_features()
logger.info("Features prepared.")

# Train and log the model
basic_model.train_model()
logger.info("Model training completed.")
basic_model.log_model()

# Register model
basic_model.register_model()

# Search for an experiment
# run_id = mlflow.search_runs(experiment_names=["/Shared/default-ccc-basic"], filter_string="tags.branch='week2'")[
#     "run_id"
# ].iloc[0]

filter_str = (
    f"tags.git_sha='{args.git_sha}' AND "
    f"tags.branch='{args.branch}' AND "
    f"tags.job_run_id='{args.job_run_id}'"
)

run_id = mlflow.search_runs(
    experiment_names=["/Shared/default-ccc-basic"], 
    filter_string=filter_str
)["run_id"].iloc[0]

model = mlflow.sklearn.load_model(model_uri=f"runs:/{run_id}/logit_pipeline_model")

# Retrieve the dataset
basic_model.retrieve_current_run_dataset()

# Retrievethe dataset metadata
basic_model.retrieve_current_run_metadata()

# Evaluate model
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100).toPandas()

model_improved = basic_model.model_improved(test_set=test_set)
logger.info("Model evaluation completed, model improved: ", model_improved)

if model_improved:
    # Register the model
    latest_version = basic_model.register_model()
    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
