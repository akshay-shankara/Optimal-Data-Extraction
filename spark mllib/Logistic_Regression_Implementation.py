from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
import sys

# Get individual from GA
individual = sys.argv[1]
array = []
individual1 = individual[0:200]
individual2 = individual[200:400]
individual3 = individual[400:600]
individual4 = individual[600:760]
individuals = [individual1, individual2, individual3]
accuracies = []
m = 0
sc = SparkContext(appName="PythonDecisionTreeClassificationExample")

for individual in individuals:
    # Filter the subset obtained from the individual
    z = 0
    while z < 200:
        if individual[z] == '0':
            array.append(0)
        else:
            array.append(1)
        z += 1

    # Set SparkContext
    i = -1
    j = -1
    # Obtain training and test files
    train_data_file = "hdfs://cshvm27:9000/akshank/data/train_data.csv"
    train_label_file = "hdfs://cshvm27:9000/akshank/data/train_labels.csv"
    test_data_file = "hdfs://cshvm27:9000/akshank/data/test_data.csv"
    test_label_file = "hdfs://cshvm27:9000/akshank/data/test_labels.csv"

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
        while column_number < 200:
            if array[m + column_number] == 1:
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
        while column_number < 200:
            if array[m + column_number] == 1:
                row.append(x[column_number])
                index += 1
            column_number += 1

        return LabeledPoint(train_labels_array[j], row)


    trainingData = train_data.map(f)
    test = test_data.map(g)

    # Train model using Logistic
    model = LogisticRegressionWithLBFGS.train(trainingData, numClasses=12613)

    predictions = model.predict(test.map(lambda x: x.features))
    labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)

    # Obtain the accuracy
    test_accuracy = labelsAndPredictions.filter(
        lambda lp: lp[0] == lp[1]).count() / float(test.count())
    accuracies.append(test_accuracy)
    m += 200

# LAST individual
individual = individual4
z = 0
while z < 160:
    if individual[z] == '0':
        array.append(0)
    else:
        array.append(1)
    z += 1

# Set SparkContext
i = -1
j = -1
# Obtain training and test files
train_data_file = "hdfs://cshvm27:9000/akshank/data/train_data.csv"
train_label_file = "hdfs://cshvm27:9000/akshank/data/train_labels.csv"
test_data_file = "hdfs://cshvm27:9000/akshank/data/test_data.csv"
test_label_file = "hdfs://cshvm27:9000/akshank/data/test_labels.csv"

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
    while column_number < 160:
        if array[m + column_number] == 1:
            row.append(x[600 + column_number])
        column_number += 1
    return LabeledPoint(train_labels_array[i], row)


# Format the test RDD
def g(x):
    global j, test_labels_array
    column_number = 0
    index = 1
    j += 1
    row = []
    while column_number < 160:
        if array[m + column_number] == 1:
            row.append(x[600 + column_number])
            index += 1
        column_number += 1

    return LabeledPoint(train_labels_array[j], row)


trainingData = train_data.map(f)
test = test_data.map(g)

# Train model using Logistic Regression
model = LogisticRegressionWithLBFGS.train(trainingData, numClasses=12613)

predictions = model.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)

# Obtain the accuracy
test_accuracy = labelsAndPredictions.filter(
    lambda lp: lp[0] == lp[1]).count() / float(test.count())
accuracies.append(test_accuracy)

test_accuracy = sum(accuracies) / float(len(accuracies))
print(str(test_accuracy))
