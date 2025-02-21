import yaml
from loguru import logger
from pyspark.sql import SparkSession

from defaultccc.config import ProjectConfig
from defaultccc.data_processor import DataProcessor

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/default_of_credit_card_clients/default_of_credit_card_clients.csv", header=True, inferSchema=True
).toPandas()

df.head(10)