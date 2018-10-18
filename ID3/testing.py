# Error tests for training and test sets

import sys

import ID3
sys.path.append('experiment-data')
import data
import functions
import numpy as np

trainingData = data.Data(fpath='experiment-data/data/train.csv')
testData = data.Data(fpath='experiment-data/data/test.csv')

trainingAttributes = dict(trainingData.column_index_dict)
del trainingAttributes['label']
trainingModel1 = ID3.ID3(trainingData, trainingAttributes)

def traverseTreeForPredictedOutcome(tree, dataColumn, dataSet):
    if tree.splitOn != None:
        attrIndex = dataSet.get_column_index(tree.splitOn)
        rowValue = dataColumn[attrIndex]
        if rowValue in tree.childNodes:
            nextTreeNode = tree.childNodes[rowValue]
            return traverseTreeForPredictedOutcome(nextTreeNode, dataColumn, dataSet)
        else:
            labels = dataSet.get_column('label')
            mostCommonValueInDataset = functions.get_common_value(labels)
            columnLabel = dataColumn[0]
            predictedCorrectly = columnLabel == mostCommonValueInDataset
            return predictedCorrectly
    else:
        columnLabel = dataColumn[0]
        predictedCorrectly = columnLabel == tree.label
        return predictedCorrectly

def getAccuracyBasedOnModel(trainingModel, dataSet):
    positiveOutcomes = 0
    negativeOutcomes = 0

    i = 0
    while i < np.size(dataSet.raw_data, 0):
        outcome = traverseTreeForPredictedOutcome(trainingModel, dataSet.raw_data[i, :], dataSet)
        if outcome:
            positiveOutcomes += 1
        else:
            negativeOutcomes += 1

        i += 1

    accuracy = positiveOutcomes/(positiveOutcomes + negativeOutcomes)
    return accuracy


print("Training Data Accuracy:", getAccuracyBasedOnModel(trainingModel1, trainingData))
print("Test Data Accuracy:", getAccuracyBasedOnModel(trainingModel1, testData))