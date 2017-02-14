# Databricks notebook source
# We can use either methods to load Iris data into Spark
# We are using the latter for better readability. Notice we have already loaded the table in Databricks, thanks to the import UI

#irisDF = sqlContext.read.format("csv").load("/FileStore/tables/15vrd1191483447849179/iris.csv", header=True)

irisDF = sqlContext.sql("SELECT * FROM iris")
display(irisDF)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

irisML = irisDF

# splitting data between training and test
(trainingData, testData) = irisML.randomSplit([0.7, 0.3])

ignore = ['species']

# assembling features
# transforming all the feature columns to one Vector column
assembler = VectorAssembler(
  inputCols=[x for x in irisML.columns if x not in ignore],
  outputCol="features")
assembled_df = assembler.transform(irisML)

# indexing label col
labelIndexer = StringIndexer(inputCol="species", outputCol="indexedLabel")
lbl_indexed_df = labelIndexer.fit(assembled_df).transform(assembled_df)

# indexing features col
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4)
ftrs_indexed_df = featureIndexer.fit(lbl_indexed_df).transform(lbl_indexed_df)

# declaring classifier
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# pipelining stages, chaining assembler and indexers
pipeline = Pipeline(stages=[assembler, labelIndexer, featureIndexer, dt])

# COMMAND ----------

# training model
model = pipeline.fit(trainingData)
# predicting values
predictions = model.transform(testData)

# COMMAND ----------

# printing wrongly predicted values
predictions.select("prediction", "indexedLabel", "features").filter(predictions.prediction != predictions.indexedLabel).show()

# measuring accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

# printing decision tree
print model.stages[3].toDebugString
