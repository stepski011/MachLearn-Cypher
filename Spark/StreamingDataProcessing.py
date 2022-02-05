from pyspark.sql.functions import *
from pyspark.sql.types import *

spark.conf.set("spark.sql.shuffle.partitions", "2")

connectionString = "Endpoint=sb://..."

ehConf = {
  'eventhubs.connectionString' : sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(connectionString),
  'eventhubs.consumerGroup' : "62"
}

stream_data_frame = spark \
  .readStream \
  .format("eventhubs") \
  .options(**ehConf) \
  .load()

body_df = stream_df.select(stream_df.body.cast('string'))
display(body_df)

# Parsing JSON values
example_location = "/FileStore/tables/batch_example.json"
example_value = spark.read.json(example_location)

json_schema = example_value.schema
batch_schema = ArrayType(json_schema)

base_df = (spark.
           readStream.
           format("eventhubs").
           options(**ehConf).
           load().
          select(
            explode(
              from_json(
                col("body").cast("string"), batch_schema)).
                   alias("message")))


json_batch_df = body_df.select(from_json('body',batch_schema ).alias('batchBody'), 'body')
display(json_batch_df)

message_df = json_batch_df.select(explode(json_batch_df.batchBody).alias("message"))
display(message_df)

country_sensorvalues_df = message_df.select(
                                      "message.location.country",
                                      "message.timestamp",
                                      "message.sensordatavalues")

count_df = country_sensorvalues_df.select("country").groupBy("country").count()

