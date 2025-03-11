import argparse

import mlflow
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

from defaultccc.config import ProjectConfig, Tags
from defaultccc.models.model_basic import BasicModel

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

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

# Evaluate model
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)

model_improved = basic_model.model_improved(test_set=test_set)
logger.info("Model evaluation completed")

if model_improved:
    # Register the model
    latest_version = basic_model.register_model()
    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)
    logger.info("Model improved!")

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
    logger.info("Model did not improved!")
