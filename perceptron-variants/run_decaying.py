import decaying_rate_perceptron as drp
import data_loading as dl
import testing as t
import numpy as np
import pdb

# Data and folds provided by class instructor, obtained from the liblinear repository, heavily modified for class purposes (the original data may not work with this code)
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#diabetes
#
# trainingData = dl.import_batch(fname="dataset/diabetes.train", n = 19)
# testingData = dl.import_batch(fname="dataset/diabetes.test", n = 19)
# devData = dl.import_batch(fname="dataset/diabetes.dev", n = 19)
# 
# folds = []
# for num in range(5):
#     folds.append(dl.import_batch(fname="dataset/CVSplits/training0" + str(num) + ".data", n = 19))

learningRates = [1, 0.1, 0.01]
epochSettings = [10, 20]
# Add Spacing
print("")
print("")
print("DECAYING RATE PERCEPTRON")
for learningRate in learningRates:
    print('-----------------------Learning Rate: ' + str(learningRate) + '----------------------')
    for epochSetting in epochSettings:
        print('-------------Epoch: ' + str(epochSetting) + '-------------')
        print('Learning Rate:', learningRate)
        print('Epochs:', str(epochSetting))
        weightVector, bias, epochData = drp.decaying_rate_perceptron(trainingData, num=19, lr=learningRate, epochs=epochSetting)
        print('Dev Majority Baseline Accuracy:', t.getMajorityBaseline(devData))
        print('Dev Accuracy:', t.testBatch(devData, weightVector, bias))
        print('Cross Validation:', t.crossValidation(drp.decaying_rate_perceptron, folds, 19, learningRate, epochSetting))
        # Add accuracies to the epoch data
        for index, epochDatum in enumerate(epochData):
            epochDatum['accuracy'] = t.testBatch(devData, epochDatum['weightObject'], epochDatum['bias'])
        print('Epoch Accuracy', dict(enumerate([epoch['accuracy'] for epoch in epochData], 1)))
        print('Epoch Learning Updates', dict(enumerate([epoch['learningUpdates'] for epoch in epochData], 1)))

        # Get the test accuracy for the best performing epoch on the development set.
        accuracyVals = [epoch['accuracy'] for epoch in epochData]
        maxIndex = accuracyVals.index(max(accuracyVals))
        print('Best epoch from dev set:', str(maxIndex + 1))
        print('Test Majority Baseline Accuracy:', t.getMajorityBaseline(testingData))
        print('Test Accuracy for best epoch:', t.testBatch(testingData, epochData[maxIndex]['weightObject'], epochData[maxIndex]['bias']))