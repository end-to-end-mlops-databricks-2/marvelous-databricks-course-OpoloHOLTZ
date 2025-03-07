import os
import time
from typing import Dict, List

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from defaultccc.config import ProjectConfig
from defaultccc.serving.model_serving import ModelServing

# spark session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{config.catalog_name}.{config.schema_name}.default_ccc_model_basic", 
    endpoint_name="default_ccc-model-serving"
)

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()

# Create a sample request body
required_columns = config.num_features + config.cat_features

# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample 100 records from the training set
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# Call the endpoint with one sample record
model_serving.call_endpoint(dataframe_records[0])

# Call the endpoint with one training set sample and evaluate
model_serving.evaluate_serving_model(test_set[required_columns + [config.target]].sample(n=100, replace=True), config.target)