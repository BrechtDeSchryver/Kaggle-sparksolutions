import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler
import numpy as np
import pandas as pd

# De spark sessie wordt aangemaakt
spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("Digits").master("local[*]").getOrCreate()


# de train en test data worden ingelezen
train_df = spark.read.csv('digit_train.csv', header=True, inferSchema=True)
test_df = spark.read.csv('digit_test.csv', header=True, inferSchema=True)

# de vector assembler wordt aangemaakt om de pixels om te zetten naar een vector
assembler = VectorAssembler(inputCols=["pixel{}".format(i) for i in range(784)], outputCol="features")

# de input en output kolommen worden ingesteld
assembler.setInputCols(["pixel{}".format(i) for i in range(784)])
assembler.setOutputCol("features")


# de vector assembler wordt toegepast op de train en test data
train_data = assembler.transform(train_df)
test_data = assembler.transform(test_df.drop("label"))


# Dit werd gebruikt om de accuracy te testen (91% accuracy)
training, val_data = train_data.randomSplit([0.8, 0.2])

# Logistiek regressie model wordt aangemaakt
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=10, regParam=0.01)
model = lr.fit(train_data)
# het model wordt getest op de test data
predictions = model.transform(test_data)
predictions = predictions.select(["prediction"])
# de predicitons worden naar een pandas dataframe omgezet
predictions = predictions.toPandas()
# de predicion wordt naar een numpy array omgezet
predictions = predictions.values


predictions = np.array(predictions)
predictions = predictions.flatten()
predictions = predictions.astype(int)

predictions_df = pd.DataFrame({"ImageId": range(1, len(predictions)+1), "Label": predictions})

# numpy array wordt naar een csv geschreven
predictions_df.to_csv("predictions.csv", index=False)

# Use the trained model to predict the labels for the test data
##predictions = model.transform(val_data)
# Select the columns containing the actual and predicted labels
##predictions = predictions.select(["label", "prediction"])
# Calculate the accuracy of the model
##accuracy = predictions.filter(predictions.label == predictions.prediction).count() / predictions.count()
# Print the accuracy
##print("Accuracy:", accuracy)