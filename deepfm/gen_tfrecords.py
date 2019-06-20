from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("deepfm_tfrecords") \
    .enableHiveSupport() \
    .getOrCreate()
all_df = spark.sql("select * from mlg.tablename")
all_df.repartition(100).write.format("tfrecords")\
    .mode("overwrite").option("recordType", "Example")\
    .save('./deepfm_data/')

