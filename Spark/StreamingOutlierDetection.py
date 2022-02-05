from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor

connectionString = "Endpoint=sb://..."
ehConf = {
  'eventhubs.connectionString' : sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(connectionString),
  'eventhubs.consumerGroup' : "79"
}

example_location = "/FileStore/tables/batch_example.json"
example_value = spark.read.json(example_location)

json_schema = example_value.schema
batch_schema = ArrayType(json_schema)

d = spark.readStream.format("eventhubs").options(**ehConf).load()
base_df = d.select(explode(from_json(col("body").cast("string"), batch_schema)).alias("message"))


# Preprocessing

training_df = base_df.select(
  explode("message.sensordatavalues").alias("sensordata"),
  base_df.message.timestamp.alias("timestamp"), 
  base_df.message.location.country.alias("country"),
  base_df.message.location.longitude.alias("longitude"),
  base_df.message.location.latitude.alias("latitude"),
  base_df.message.location.altitude.alias("altitude"),
  base_df.message.location.id.alias("location_id")
).select(
  "timestamp", "country", "latitude", "longitude", "sensordata.id", "sensordata.value", "sensordata.value_type","location_id"
)

training_df = training_df.withColumn("timestamp_format", to_timestamp("timestamp"))
training_df = training_df.withColumn("dayofyear", dayofyear("timestamp_format"))
training_df = training_df.withColumn("hourofday", hour("timestamp_format"))
training_df = training_df.withColumn("minuteofhour", minute("timestamp_format"))

training_df = training_df.withColumn("temperature", when(col("value_type") == "temperature",col("value")).otherwise(-999.0).astype("int"))
training_df = training_df.withColumn("humidity", when(col("value_type") == "humidity",col("value")).otherwise(-999.0).astype("int"))
training_df = training_df.withColumn("P1", when(col("value_type") == "P1",col("value")).otherwise(-999.0).astype("int"))
training_df = training_df.withColumn("P2", when(col("value_type") == "P2",col("value")).otherwise(-999.0).astype("int"))

live_df = training_df.groupby(
  "location_id", "dayofyear", "hourofday", "minuteofhour"
).agg(
  max("temperature").alias("temperature"), 
  max("humidity").alias("humidity"), 
  max("P1").alias("P1"),
  max("P2").alias("P2"))\
.filter(col("temperature") != -999.0)\
.filter(col("humidity") != -999.0)\
.filter(col("P1") != -999.0)\
.filter(col("P2") != -999.0)

display(live_df)


# K-Means
assembler = VectorAssembler().setInputCols(['P1', 'P2', 'humidity', 'temperature']).setOutputCol('features')

kmeans = KMeans(k=7, distanceMeasure='euclidean').setSeed(42)

df_train = spark.read.parquet("/FileStore/lsf/df_train.parquet")
kMeansModel = kmeans.fit(assembler.transform(df_train))

for clusterID, center in enumerate(kMeansModel.clusterCenters()):
  print(f"cluster id: {clusterID}, cluster centers: {center}")

main_cluster = kMeansModel.clusterCenters()[0]
print(main_cluster[0])

schema = (StructType([StructField("P1", DoubleType(), True), 
                      StructField("P2", DoubleType(), True),
                      StructField("humidity", DoubleType(), True),
                      StructField("temperature", DoubleType(), True)]))

streamingData = (spark.readStream 
                 .schema(schema) 
                 .option("maxFilesPerTrigger", 1) 
                 .parquet("/FileStore/lsf/outlier_detection_test_data"))

# Plotting outliers

display(
  kMeansModel.transform(
    assembler.transform(live_df)
  ).filter(
    col("prediction")!=0)
)

# Showing overall class counts on local streaming data
display(
  kMeansModel.transform(
    assembler.transform(
      streamingData)
  ).groupby("prediction").count()
)

# Visualizing clusters in a scatter plot 
display(
  kMeansModel.transform(
    assembler.transform(streamingData)
  ).select(
    log("temperature"), 
    log("P2"), 
    "P2", 
    "temperature", 
    "humidity", 
    log("humidity"), 
    "P1", 
    "prediction")
)

# Gradient Boosted Tree

vectorAssembler = VectorAssembler(inputCols=["P1","humidity","temperature"], outputCol="features")

estimator = GBTRegressor(labelCol="P2", featuresCol="features", maxDepth=3, maxBins=32, maxIter=10)

pipeline = Pipeline(stages=[vectorAssembler, estimator])

df_train = spark.read.parquet("/FileStore/lsf/df_train.parquet")
pipelineModel = pipeline.fit(df_train)

# Predictions
display(
  pipelineModel.transform(
    streamingData
  ).select(
    "P2", "prediction", 
    abs(col("P2") - col("prediction")).alias("diff")
  )
)

#RMSE
display(
  pipelineModel.transform(
    streamingData
  ).agg(
    sqrt(avg(pow(col("P2") - col("prediction"), 2))).alias("root mean square error")
  )
)

# Predictions on cleaned data (model was not trained on cleaned data)
display(
  pipelineModel.transform(
    kMeansModel.transform(
      assembler.transform(
        streamingData
      )
    )
    .filter(col("prediction")==0).select(
      "P1", "P2", "humidity", "temperature")
  ).select(
    "P2", "prediction", 
    abs(col("P2") - col("prediction")).alias("diff")
  )
)

# Rmse
display(
  pipelineModel.transform(
    kMeansModel.transform(
      assembler.transform(
        streamingData
      )
    ).filter(
      col("prediction")==0
    ).select(
      "P1", "P2", "humidity", "temperature"
    )
  ).agg(
    sqrt(avg(pow(col("P2") - col("prediction"), 2))).alias("root mean square error")
  )
)

# Retraining the model on cleaned data
pipelineModelRetrained = pipeline.fit(    
  kMeansModel.transform(
      assembler.transform(
        df_train
      )).filter(col("prediction")==0)
  .select(
    "P1", "P2", "humidity", "temperature"
  )
)

# New model
display(
  pipelineModelRetrained.transform(
    kMeansModel.transform(
      assembler.transform(
        streamingData
      )
    )
    .filter(col("prediction")==0).select(
      "P1", "P2", "humidity", "temperature")
  ).select(
    "P2", "prediction", 
    abs(col("P2") - col("prediction")).alias("diff")
  )
)

# Rmse
display(
  pipelineModelRetrained.transform(
    kMeansModel.transform(
      assembler.transform(
        streamingData
      )
    ).filter(
      col("prediction")==0
    ).select(
      "P1", "P2", "humidity", "temperature"
    )
  ).agg(
    sqrt(avg(pow(col("P2") - col("prediction"), 2))).alias("root mean square error")
  )
)