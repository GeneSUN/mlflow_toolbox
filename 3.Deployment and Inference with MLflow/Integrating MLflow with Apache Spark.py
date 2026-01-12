
import mlflow
logged_model = 'runs:/6815b44128e14df2b356c9db23b7f936/model'
df = spark.read.format("csv").load("dbfs:/FileStore/shared_uploads/ input.csv")

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# Predict on a Spark DataFrame.
df.withColumn('predictions', loaded_model()).collect()
