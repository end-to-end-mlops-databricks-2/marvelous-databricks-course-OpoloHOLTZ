import yaml
from loguru import logger

from defaultccc.config import ProjectConfig
from defaultccc.data_processor import DataProcessor

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    path=f"/Volumes/{config.catalog_name}/{config.schema_name}/default_of_credit_card_clients/default_of_credit_card_clients.csv",
    header=True,
    inferSchema=True,
    sep=";",
).toPandas()

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info(f"Training set shape: {X_train.shape}")
logger.info(f"Test set shape: {X_test.shape}")

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
