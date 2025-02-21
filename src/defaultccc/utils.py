import pyspark

def initialize_spark() -> pyspark.sql.SparkSession:
    """Initialize Spark session in development environment.
 
    Returns
    -------
        Initialized Spark session.
 
    """
    # load_dotenv(find_dotenv())
 
    builder = (
        pyspark.sql.SparkSession.builder.appName("PoCSerbia")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-azure:2.7.2")
        .config("spark.driver.memory", "1g")
    )
 
    session = builder.getOrCreate()
 
    # logger.info("Initialized spark session")
    return session