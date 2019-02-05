import pdb
import numpy as np
import utility_functions as uf
from multiprocessing import Process, Queue


def testBatch(batch, model, num):
    TP = 0
    FP = 0
    FN = 0
    accuracySum = 0

    # If there is a transformation to be made to the feature space, make it
    if hasattr(model, 'transform_testing_space'):
        batch = model.transform_testing_space(batch, num)

    for batchObj in batch:
        prediction = model.predict(batchObj['instances'])
        Y = batchObj['label']
        if prediction == 1 and Y == 1:
            TP += 1
        elif prediction == 1 and Y == -1:
            FP += 1
        elif prediction == -1 and Y == 1:
            FN += 1

        if prediction == Y:
            accuracySum += 1

    P = TP/((TP + FP) or 1)
    R = TP/((TP + FN) or 1)
    F = 2 * ((P * R)/((P + R) or 1))
    acc = accuracySum/len(batch)
    
    return F, P, R, acc

def crossValidation(model, folds, num, smooth_by=None, lr=None, C=None, epochs=None, depth=None):
    # Do a test for each variation of folds
    metrics = []
    processes = []
    queue = Queue()
    for index, testFold in enumerate(folds):
        process = Process(target=get_fold_metrics, args=(model, folds, num, len(folds), index), kwargs={'smooth_by': smooth_by, 'lr': lr, 'C' :C, 'epochs': epochs, 'depth': depth, 'queue': queue})
        processes.append(process)
        process.start()

    for process in processes:
        metrics.append(queue.get())
    
    for process in processes:
        process.join()


    numFolds = len(folds)
    averageFScore = sum([metric[0] for metric in metrics])/numFolds
    averageP = sum([metric[1] for metric in metrics])/numFolds
    averageR = sum([metric[2] for metric in metrics])/numFolds
    averageAcc = sum([metric[3] for metric in metrics])/numFolds

    return averageFScore, averageP, averageR, averageAcc


def get_fold_metrics(model, folds, num, foldLen, index, smooth_by=None, lr=None, C=None, epochs=None, depth=None, queue=None):
    # Go through folds, concat them if they do not match the current fold's index
    concattedFolds = []
    for ind in range(foldLen):
        if ind != index:
            concattedFolds += folds[ind]
    model_instance = model()
    if lr and C and epochs and depth:
        model_instance.train(concattedFolds, num, lr=lr, epochs=epochs, C=C, depth=depth)
    elif lr and C and epochs:
        model_instance.train(concattedFolds, num, lr=lr, epochs=epochs, C=C)
    elif smooth_by:
        model_instance.train(concattedFolds, num, smooth_by=smooth_by)
    F, P, R, acc = testBatch(concattedFolds, model_instance, num)
    if queue:
        queue.put((F, P, R, acc))
    return(F, P, R, acc)