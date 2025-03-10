import mlflow
from pyspark.sql import SparkSession

from defaultccc.config import ProjectConfig, Tags
from defaultccc.models.model_basic import BasicModel

mlflow.set_tracking_uri("databricks://opoloholtz")
mlflow.set_registry_uri("databricks-uc://opoloholtz")

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
run_id = mlflow.search_runs(experiment_names=["/Shared/default-ccc-basic"], filter_string="tags.branch='week2'")[
    "run_id"
].iloc[0]

model = mlflow.sklearn.load_model(model_uri=f"runs:/{run_id}/logit_pipeline_model")

# Retrieve the dataset
basic_model.retrieve_current_run_dataset()

# Retrievethe dataset metadata
basic_model.retrieve_current_run_metadata()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

prediction_df = basic_model.load_latest_model_and_predict(
    test_set[test_set.columns[~test_set.columns.isin([config.target])]]
)

print(prediction_df)
