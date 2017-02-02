# Databricks notebook source
# MAGIC %md
# MAGIC #Importing data
# MAGIC We first need to import the data from the San Francisco Open Data website.
# MAGIC 
# MAGIC Dataset can be found at https://data.sfgov.org/Public-Safety/SFPD-Incidents-from-1-January-2003/tmnf-yvry#

# COMMAND ----------

import base64
import urllib
import json

url = "https://data.sfgov.org/resource/cuks-n6tp.json"

response = urllib.urlopen(url)
raw_data = json.loads(response.read().decode('utf-8'))

# COMMAND ----------

from pyspark.sql.types import *

raw_df = sqlContext.createDataFrame(raw_data)

dropped_df = (raw_df.drop(':@computed_region_bh8s_q3mv')
            .drop(':@computed_region_fyvs_ahh9')
            .drop(':@computed_region_p5aj_wyqh')
            .drop(':@computed_region_rxqg_mtj9')
            .drop(':@computed_region_yftq_j783'))

df = (dropped_df
      .withColumn('x', dropped_df['x'].cast(IntegerType()))
      .withColumn('y', dropped_df['y'].cast(IntegerType()))
      .withColumn('pdid', dropped_df['pdid'].cast(IntegerType()))
      .withColumn('date', dropped_df['date'].cast(DateType())))

# COMMAND ----------

display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Analysis
# MAGIC Now that we have imported the data into a Spark DataFrame, we can analyze it

# COMMAND ----------

df.select('category').groupby('category').count().orderBy("count", ascending=False).show()

# COMMAND ----------

df.select('date').groupby('date').count().orderBy("date").show()

# COMMAND ----------

df.repartition(4).createOrReplaceTempView("df_view")
spark.catalog.cacheTable("df_view")
spark.table("df_view").write.format('parquet').mode('overwrite').save('/tmp/crimeParquet')

# COMMAND ----------

# MAGIC %fs ls /tmp/crimeParquet/

# COMMAND ----------

tempDF = spark.read.parquet('/tmp/crimeParquet')
display(tempDF)

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT DISTINCT descript, COUNT(*)
# MAGIC FROM df_view
# MAGIC GROUP BY descript
# MAGIC ORDER BY 2 DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT PDDISTRICT, DESCRIPT, COUNT(*)
# MAGIC FROM df_view
# MAGIC GROUP BY PDDISTRICT, DESCRIPT
# MAGIC ORDER BY 3 DESC
