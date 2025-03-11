import argparse

import yaml
from loguru import logger
from pyspark.sql import SparkSession

from defaultccc.config import ProjectConfig
from defaultccc.data_processor import DataProcessor, generate_synthetic_data

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--root_path",
#     action="store",
#     default="/Workspace/Users/opolo.holtz@amaris.com/.bundle/marvelous-databricks-course-OpoloHOLTZ",
#     type=str
# )

# parser.add_argument(
#     "--env",
#     action="store",
#     default="dev",
#     type=str
# )
# args = parser.parse_args()

args = argparse.Namespace(
    root_path="/Workspace/Users/opolo.holtz@amaris.com/.bundle/marvelous-databricks-course-OpoloHOLTZ/dev",
    env="dev"
)

root_path = args.root_path
# config_path = f"{root_path}/files/project_config.yml"
config_path = "../project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    path=f"/Volumes/{config.catalog_name}/{config.schema_name}/default_of_credit_card_clients/default_of_credit_card_clients.csv",
    header=True,
    inferSchema=True,
    sep=";",
).toPandas()

# Generate synthetic data
synthetic_df = generate_synthetic_data(df, num_rows=100)
logger.info("Synthetic data generated.")

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info(f"Training set shape: {X_train.shape}")
logger.info(f"Test set shape: {X_test.shape}")

print(X_train)
print(X_test)

# Save to catalog
logger.info("Saving data to catalog.")
data_processor.save_to_catalog(X_train, X_test)
