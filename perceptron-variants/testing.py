import numpy as np
import simple_perceptron as sp
import functions as f
import pdb

def testBatch(batch, W, bias, margin = 0):
    correct = 0
    incorrect = 0
    for batchObj in batch:
        XT = np.transpose(batchObj['dataArray'])
        Y = batchObj['label']
        mistake = Y * (np.sum(W * XT) + bias) < margin
        if mistake:
            incorrect += 1
        else:
            correct += 1
    return correct/(correct + incorrect)

def crossValidation(algorithm, folds, num, lr, epochs, margin = 0):
    accuracySum = 0
    foldLen = len(folds)
    # Do a test for each variation of folds
    for index, testFold in enumerate(folds):
        # Go through folds, concat them if they do not match the current fold's index
        concattedFolds = []
        for ind in range(foldLen):
            if ind != index:
                concattedFolds += folds[ind]
        if margin > 0:
            weightVector, bias, epochData = algorithm(concattedFolds, num=num, lr=lr, epochs=epochs, margin=margin)
        else:
            weightVector, bias, epochData = algorithm(concattedFolds, num=num, lr=lr, epochs=epochs)
        testFoldAccuracy = testBatch(concattedFolds, weightVector, bias, margin=margin)
        accuracySum += testFoldAccuracy
    return accuracySum/foldLen

def getMajorityBaseline(batch):
    positiveLabels = 0
    negativeLabels = 0
    majorityLabel = 0
    for XObj in batch:
        if XObj['label'] == 1.0:
            positiveLabels += 1
        else:
            negativeLabels += 1

    majorityLabel = positiveLabels if positiveLabels > negativeLabels else negativeLabels
    return majorityLabel/(positiveLabels + negativeLabels)

