import sys
import ID3
import testing
import data
import numpy as np


# Data and folds provided by class instructor, obtained from the liblinear repository, heavily modified for class purposes (the original data will not work with this code)
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#mushrooms
#
# trainingData = data.Data(fpath='experiment-data/data/train.csv')
# testData = data.Data(fpath='experiment-data/data/test.csv')
# fold1 = np.loadtxt(fname='experiment-data/data/CVfolds/fold1.csv', delimiter=',', dtype = str, skiprows=1)
# fold2 = np.loadtxt(fname='experiment-data/data/CVfolds/fold2.csv', delimiter=',', dtype = str, skiprows=1)
# fold3 = np.loadtxt(fname='experiment-data/data/CVfolds/fold3.csv', delimiter=',', dtype = str, skiprows=1)
# fold4 = np.loadtxt(fname='experiment-data/data/CVfolds/fold4.csv', delimiter=',', dtype = str, skiprows=1)
# fold5 = np.loadtxt(fname='experiment-data/data/CVfolds/fold5.csv', delimiter=',', dtype = str, skiprows=1)
# foldForAttributes = np.loadtxt(fname='experiment-data/data/CVfolds/fold1.csv', delimiter=',', dtype = str)


attributes = foldForAttributes[0]
attributesWithoutLabel = attributes[1:]
attributesWithoutLabel = dict.fromkeys(attributesWithoutLabel, 0)

def crossfoldValidation(seqOfData, hyperparameter = None):
    accuracyValues = []
    count = 0
    while count < len(seqOfData):
        dataToConcat = seqOfData.copy()
        # Withheld fold to validate on
        dataWithheld = dataToConcat[count]
        # insert the attributes back in
        dataWithheld = np.vstack([attributes, dataWithheld])
        # now make it a data class
        dataWithheld = data.Data( data = dataWithheld )
        
        # Concatenation of other folds
        del dataToConcat[count]
        dataToModel = np.concatenate(dataToConcat)
        # insert the attributes
        dataToModel = np.vstack([attributes, dataToModel])

        trainingSet = data.Data( data = dataToModel )
        # create model on the concatenated 4 folds
        trainingModel = ID3.ID3(trainingSet, attributesWithoutLabel, hyperparameter = hyperparameter)
        accuracy = testing.getAccuracyBasedOnModel(trainingModel, dataWithheld)
        accuracyValues.append(accuracy)
        count += 1

    accuracyAverage = np.average(accuracyValues)
    accuracyStandardDeviation = np.std(accuracyValues)

    return accuracyAverage, accuracyStandardDeviation

hyperparameters = [1, 2, 3, 4, 5, 10, 15]

for h in hyperparameters:
    avg, std = crossfoldValidation([fold1, fold2, fold3, fold4, fold5], hyperparameter = h)
    print('Hyperparameter =', str(h), ', average:', avg, ', standard deviation:', std)   

# Test the accuracy of hyperparameter = 5 on the full training and test sets
trainingModelAt5 = ID3.ID3(trainingData, attributesWithoutLabel, "root", hyperparameter = 5)
testingSetAccuracy = testing.getAccuracyBasedOnModel(trainingModelAt5, testData)
print('Testing Set Accuracy:', testingSetAccuracy)