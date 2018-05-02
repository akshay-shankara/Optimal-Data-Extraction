from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark.mllib.tree import DecisionTree
import sys

# Get individual from GA
individual = sys.argv[1]
array = []
z = 0

# Filter the subset obtained from the individual
while z < 760:
    if individual[z] == '0':
        array.append(0)
    else:
        array.append(1)
    z += 1

# Set SparkContext
sc = SparkContext(appName="PythonDecisionTreeClassificationExample")
i = -1
j = -1

# Obtain training and test files
train_data_file = "./data/train_data.csv"
train_label_file = "./data/train_labels.csv"
test_data_file = "./data/test_data.csv"
test_label_file = "./data/test_labels.csv"

# Set the RDDs for test and training data
train_data = sc.textFile(train_data_file)
train_data = train_data.map(lambda x: x.split(","))
test_data = sc.textFile(test_data_file)
test_data = test_data.map(lambda x: x.split(","))

# Set the RDDs for test and training labels
train_labels = sc.textFile(train_label_file)
train_labels_array = train_labels.collect()
test_labels = sc.textFile(test_label_file)
test_labels_array = test_labels.collect()


# Format the train RDD
def f(x):
    global i, train_labels_array
    column_number = 0
    row = []
    i += 1
    while column_number < 760:
        if array[column_number] == 1:
            row.append(x[column_number])
        column_number += 1
    return LabeledPoint(train_labels_array[i], row)


# Format the test RDD
def g(x):
    global j, test_labels_array
    column_number = 0
    index = 1
    j += 1
    row = []
    while column_number < 760:
        if array[column_number] == 1:
            row[index] = x[column_number]
            index += 1
        column_number += 1

    return LabeledPoint(train_labels_array[j], row)


trainingData = train_data.map(f)
test = test_data.map(g)

# Train model using DECISION TREE
model = DecisionTree.trainClassifier(trainingData, numClasses=12609, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=2, maxBins=32)

predictions = model.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)

# Obtain the accuracy
test_accuracy = labelsAndPredictions.filter(
    lambda lp: lp[0] == lp[1]).count() / float(test.count())
str(test_accuracy)
