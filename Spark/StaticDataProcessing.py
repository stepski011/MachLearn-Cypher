from pyspark.sql.functions import *
from pyspark.sql.types import *

%fs ls /source/parquet/2019/07-09_dht22_muenster_hannover/
%fs ls /source/parquet/2019/07-09_dht22_muenster_hannover/

parquet_data_frame = spark.read.parquet("/source/parquet/2019/07-09_dht22_muenster_hannover/")
parquetDF = spark.read.parquet("/source/parquet/2019/07-09_dht22_muenster_hannover/")
display(parquet_data_frame)

parquet_data_frame.createOrReplaceTempView("tmp_dht22")
parquetDF.createOrReplaceTempView("tmp_dht22")

%sql 

Select * from tmp_dht22;
create database created_group;
--drop database group99
create database group99;
create table grupa_dht22 as (select * from tmp_dht22);
create table group99.dht22 as (select * from tmp_dht22);
--drop table group99.dht22;
parquetDF.write.saveAsTable("group99.dht22")

%sql
select 
  city, 
  avg(temperature), 
  avg(humidity), 
  window(timestamp, "1 day")["start"] as ts_window 
from 
  group99.dht22
group by 
  city, 
  ts_window;

# Aggregation for 1 day
df_window = (parquet_data_frame.select("city", "sensor_Type", "temperature", "humidity", window("timestamp","1 day").alias("ts_window"))
            .groupby("city", "ts_window")
            .agg(avg("temperature").alias("avg_temperature"), avg("humidity").alias("avg_humidity")).orderBy("city","ts_window.start")
            )
display(df_window)
display(tmp_dht22)

# Aggregation for 1 hour
df_window_air = (parquet_data_frame.select("city", "sensor_Type", "temperature", "humidity", window("timestamp","1 hour").alias("ts_window"))
            .groupby("city", "ts_window")
            .agg(avg("temperature").alias("avg_temperature"), avg("humidity").alias("avg_humidity")).orderBy("city","ts_window.start")
            )
display(df_window)