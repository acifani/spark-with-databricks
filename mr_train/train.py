# Databricks notebook source
import urllib

tracks_resp = urllib.urlopen('https://www.mapr.com/ebooks/spark/data/tracks.csv')
customers_resp = urllib.urlopen('https://www.mapr.com/ebooks/spark/data/cust.csv')

# COMMAND ----------

dbutils.fs.mkdirs("/tmp/tracks")
dbutils.fs.put("/tmp/tracks/tracks.csv", tracks_resp.read(), overwrite=True)
dbutils.fs.put("/tmp/tracks/customers.csv", customers_resp.read(), overwrite=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Files format
# MAGIC 
# MAGIC #### Tracks dataset
# MAGIC | Field Name | Event ID | Customer ID | Track ID | Datetime            | Mobile  | Listening Zip |
# MAGIC |------------|----------|-------------|----------|---------------------|---------|---------------|
# MAGIC | **Type**   | Integer  | Integer     | Integer  | String              | Integer | Integer       |
# MAGIC | **Example**| 9999767  | 2597        | 788      | 2014-12-01 09:54:09 | 0       | 11003         |
# MAGIC 
# MAGIC #### Customers dataset
# MAGIC 
# MAGIC | Field Name | Customer ID | Name              | Gender  | Address              | Zip     | Sign Date  | Status  | Level   | Campaign | Linked with apps? |
# MAGIC |------------|-------------|-------------------|---------|----------------------|---------|------------|---------|---------|----------|-------------------|
# MAGIC | **Type**   | Integer     | String            | Integer | String               | Integer | String     | Integer | Integer | Integer  | Integer           |
# MAGIC | **Example**| 10          | Joshua Threadgill | 0       | 10084 Easy Gate Bend | 66216   | 01/13/2013 | 0       | 1       | 1        | 1                 |

# COMMAND ----------

from pyspark.sql.types import *

tracks_schema = StructType([ \
    StructField("event_id", IntegerType(), True), \
    StructField("customer_id", IntegerType(), True), \
    StructField("track_id", IntegerType(), True), \
    StructField("datetime", DateType(), True), \
    StructField("mobile", IntegerType(), True), \
    StructField("listening_zip", IntegerType(), True)])

tracks_df = (sqlContext
              .read
              .format('com.databricks.spark.csv')
              .load('/tmp/tracks/tracks.csv', schema=tracks_schema))

customers_schema = StructType([ \
    StructField("customer_id", IntegerType(), True), \
    StructField("name", StringType(), True), \
    StructField("gender", IntegerType(), True), \
    StructField("address", StringType(), True), \
    StructField("zip", IntegerType(), True), \
    StructField("sign_date", StringType(), True), \
    StructField("status", IntegerType(), True), \
    StructField("level", IntegerType(), True), \
    StructField("campaign", IntegerType(), True), \
    StructField("linked_with_apps", IntegerType(), True)])

customers_df = (sqlContext
                 .read
                 .format('com.databricks.spark.csv')
                 .options(header='true')
                 .load('/tmp/tracks/customers.csv', schema=customers_schema))

# COMMAND ----------

trackfile = sc.textFile('/tmp/tracks/tracks.csv')

def map_to_keyvalue(str):
  r = str.split(",")
  # key: customer_id
  # value: array of songs listened by the customer
  return [r[1], [[int(r[2]), r[3], int(r[4]), r[5]]]]

tbycust = (trackfile
            .map(map_to_keyvalue)
            .reduceByKey(lambda x, y: x + y))

# COMMAND ----------

def compute_stats(tracks):
  mcount = morn = aft = eve = night = 0
  tracklist = []
  
  for t in tracks:
    trackid, dtime, mobile, zip = t
    
    # list of tracks listened to
    if trackid not in tracklist:
      tracklist.append(trackid)
    
    # count tracks listened in different part of day
    date, time = dtime.split(" ")
    hourofday = int(time.split(":")[0])
    if (hourofday < 5):
      night += 1
    elif (hourofday < 12):
      morn += 1
    elif (hourofday < 17):
      aft += 1
    elif (hourofday < 22):
      eve += 1
    else:
      night += 1

    # count tracks listened on mobile
    mcount += mobile
    
    return [len(tracklist), morn, aft, eve, night, mcount]
  
custdata = tbycust.mapValues(compute_stats)

from pyspark.mllib.stat import Statistics
aggdata = Statistics.colStats(custdata.map(lambda x : x[1]))

# COMMAND ----------

cust_kv = custdata.toDF()

# COMMAND ----------

unique, morn, aft, eve, night, mobile
