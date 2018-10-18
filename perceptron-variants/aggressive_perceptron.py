import numpy as np
import functions as f
import random

# XBatch is the batch of training examples
# num is the dimensionality of the features
# lr is the learning rate hyperparameter
# epochs is the number of runs the algorithm will make on the data, shuffling after each one
# margin is the hyperplane margin hyperparameter
def aggressive_perceptron(XBatch, num = 0, lr = 1, epochs = 1, margin = 0):
    # First, get random float between -0.01 and 0.01:
    randFloat = np.random.uniform(-0.01, 0.01)
    # Initialize weight vector and bias term
    W = np.full_like(np.zeros((num,1), dtype='f4'), randFloat)
    bias = randFloat
    # Initiate list to store the weight vector after each epoch
    epochDataList = []
    for epoch in range(epochs):
        # Count number of updates
        learningUpdates = 0
        for XObj in XBatch:
            # Make the prediction
            XT = np.transpose(XObj['dataArray'])
            # Real Label revealed
            dataLabel = XObj["label"]
            # Is this a mistake?
            mistakeMade = dataLabel * (np.sum(XT * W) + bias) < margin
            # If mistake was made, update W
            if mistakeMade:
                aggressiveLR = (margin - (dataLabel * np.sum(XT * W)))/((np.sum(XT * XT)) + 1)
                W = W + aggressiveLR * (dataLabel * XT)
                bias = bias + (aggressiveLR * dataLabel)
                learningUpdates += 1

        # Add the data for this epoch
        epochDataList.append({
            'weightObject': np.copy(W),
            'bias': bias,
            'learningUpdates': learningUpdates
        })
        
        # Shuffle data for next epoch
        random.shuffle(XBatch)

    return W, bias, epochDataList