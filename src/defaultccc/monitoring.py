from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType
from pyspark.sql.window import Window


def create_or_refresh_monitoring(config, spark, workspace):
    inf_table = spark.sql(
        f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`default_ccc-model-serving_payload_payload`"
    )

    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("LIMIT_BAL", IntegerType(), True),
                            StructField("BILL_AMT1", IntegerType(), True),
                            StructField("BILL_AMT2", IntegerType(), True),
                            StructField("BILL_AMT3", IntegerType(), True),
                            StructField("BILL_AMT4", IntegerType(), True),
                            StructField("BILL_AMT5", IntegerType(), True),
                            StructField("BILL_AMT6", IntegerType(), True),
                            StructField("PAY_AMT1", IntegerType(), True),
                            StructField("PAY_AMT2", IntegerType(), True),
                            StructField("PAY_AMT3", IntegerType(), True),
                            StructField("PAY_AMT4", IntegerType(), True),
                            StructField("PAY_AMT5", IntegerType(), True),
                            StructField("PAY_AMT6", IntegerType(), True),
                            StructField("AGE", StringType(), True),
                            StructField("SEX", StringType(), True),
                            StructField("EDUCATION", StringType(), True),
                            StructField("MARRIAGE", StringType(), True),
                            StructField("PAY_0", StringType(), True),
                            StructField("PAY_2", StringType(), True),
                            StructField("PAY_3", StringType(), True),
                            StructField("PAY_4", StringType(), True),
                            StructField("PAY_5", StringType(), True),
                            StructField("PAY_6", StringType(), True),
                            StructField("ID", StringType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    response_schema = StructType(
        [
            StructField("predictions", ArrayType(DoubleType()), True),
            StructField(
                "databricks_output",
                StructType(
                    [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
                ),
                True,
            ),
        ]
    )

    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema)).withColumn(
        "parsed_response", F.from_json(F.col("response"), response_schema)
    )

    df_exploded = (
        inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))
        .withColumn("row_index", F.monotonically_increasing_id())
        .withColumn("prediction", F.col("parsed_response.predictions")[F.col("row_index")])
    )

    df_final = df_exploded.select(
        F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
        "timestamp_ms",
        "databricks_request_id",
        "execution_time_ms",
        F.col("record.LIMIT_BAL").alias("LIMIT_BAL"),
        F.col("record.BILL_AMT1").alias("BILL_AMT1"),
        F.col("record.BILL_AMT2").alias("BILL_AMT2"),
        F.col("record.BILL_AMT3").alias("BILL_AMT3"),
        F.col("record.BILL_AMT4").alias("BILL_AMT4"),
        F.col("record.BILL_AMT5").alias("BILL_AMT5"),
        F.col("record.BILL_AMT6").alias("BILL_AMT6"),
        F.col("record.PAY_AMT1").alias("PAY_AMT1"),
        F.col("record.PAY_AMT2").alias("PAY_AMT2"),
        F.col("record.PAY_AMT3").alias("PAY_AMT3"),
        F.col("record.PAY_AMT4").alias("PAY_AMT4"),
        F.col("record.PAY_AMT5").alias("PAY_AMT5"),
        F.col("record.PAY_AMT6").alias("PAY_AMT6"),
        F.col("record.AGE").alias("AGE"),
        F.col("record.SEX").alias("SEX"),
        F.col("record.EDUCATION").alias("EDUCATION"),
        F.col("record.MARRIAGE").alias("MARRIAGE"),
        F.col("record.PAY_0").alias("PAY_0"),
        F.col("record.PAY_2").alias("PAY_2"),
        F.col("record.PAY_3").alias("PAY_3"),
        F.col("record.PAY_4").alias("PAY_4"),
        F.col("record.PAY_5").alias("PAY_5"),
        F.col("record.PAY_6").alias("PAY_6"),
        F.col("record.ID").alias("ID"),
        F.col("prediction").alias("prediction"),
        F.lit("default_ccc_model_basic").alias("model_name"),
    )

    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")

    window_update_timestamp = Window.partitionBy("ID").orderBy(F.col("update_timestamp_utc").desc())

    test_set_unique = (
        test_set.withColumn("row_number", F.row_number().over(window_update_timestamp))
        .filter(F.col("row_number") == 1)
        .drop("row_number")
    )

    df_final_with_status = (
        df_final.join(test_set_unique.select("ID", "default_payment_next_month"), on="ID", how="left")
        .filter(F.col("default_payment_next_month").isNotNull())
        .withColumn("default_payment_next_month", F.col("default_payment_next_month").cast("int"))
        .withColumn("prediction", F.col("prediction").cast("int"))
        .dropna(subset=["default_payment_next_month", "prediction"])
    )

    df_final_with_status.write.format("delta").mode("overwrite").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.model_monitoring"
    )

    try:
        workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
        workspace.quality_monitors.run_refresh(
            table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
        )
        logger.info("Lakehouse monitoring table exist, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table is created.")


def create_monitoring_table(config, spark, workspace):
    logger.info("Creating new monitoring table..")

    monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=f"{config.catalog_name}.{config.schema_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="default_payment_next_month",
        ),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
