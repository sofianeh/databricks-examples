# Databricks notebook source
file = open("/dbfs/myentities.txt","w") 
 
file.write("Hello World\n") 
file.write("Entity to be found\n") 
file.write("Yet another entity") 
 
file.close() 

file = open("/dbfs/myentities.txt", "r") 
print(file.read())
file.close()

# COMMAND ----------

l = [
  (1,'My first line has Hello World in it. Can it be found?'),
  (2,'My second line doesn\'t')
]

data = spark.createDataFrame(l, ['docID','text'])
display(data)

# COMMAND ----------

from pyspark.ml import Pipeline

# spark-nlp-1.3.0.jar is attached to the cluster. This library was downloaded from the
# spark-packages repository https://spark-packages.org/package/JohnSnowLabs/spark-nlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentenceDetector = SentenceDetector()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")
  
tokenizer = Tokenizer()\
  .setInputCols(["document"])\
  .setOutputCol("token")

# This should be optional as it is a workaround in this case because of the entities file content
# normalization (Read more: https://github.com/JohnSnowLabs/spark-nlp/issues/57#issuecomment-361993477)
normalizer = Normalizer()\
  .setInputCols(["token"])\
  .setOutputCol("normal")
  
extractor = EntityExtractor()\
  .setEntitiesPath("/dbfs/myentities.txt")\
  .setInputCols(["normal", "sentence"])\
  .setInsideSentences(False)\
  .setOutputCol("entities")

finisher = Finisher() \
    .setInputCols(["entities"]) \
    .setCleanAnnotations(False)\
    .setIncludeKeys(True)

pipeline = Pipeline(
    stages = [
    documentAssembler,
    sentenceDetector,
    tokenizer,
    normalizer,
    extractor,
    finisher
  ])


# COMMAND ----------

model = pipeline.fit(data)

extracted = model.transform(data)
display(extracted)