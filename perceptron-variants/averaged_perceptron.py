import numpy as np
import functions as f
import random
import pdb

# XBatch is the batch of training examples
# num is the dimensionality of the features
# lr is the learning rate hyperparameter
# epochs is the number of runs the algorithm will make on the data, shuffling after each one

def averaged_perceptron(XBatch, num = 0, lr = 1, epochs = 1):
    # First, get random float between -0.01 and 0.01:
    randFloat = np.random.uniform(-0.01, 0.01)
    # Initialize weight vector and bias term
    W = np.full_like(np.zeros((num,1), dtype='f4'), randFloat)
    bias = randFloat
    # Initialize the averaged weight vector and bias
    AW = np.copy(W)
    biasA = bias
    # Initiate list to store the weight vector after each epoch
    epochDataList = []
    # Run through every epoch
    for epoch in range(epochs):
        # Count number of updates
        learningUpdates = 0
        for XObj in XBatch:
            # Make the prediction
            XT = np.transpose(XObj['dataArray'])
            # Real Label revealed
            dataLabel = XObj["label"]
            # Is this a mistake?
            mistakeMade = dataLabel * (np.sum(XT * W) + bias) < 0
            # If mistake was made, update W
            if mistakeMade:
                W = W + lr * (dataLabel * XT)
                bias = bias + (lr * dataLabel)
                learningUpdates += 1
            # Update averaged vector and bias
            AW = AW + W
            biasA = biasA + bias

        # Add the data for this epoch
        epochDataList.append({
            'weightObject': np.copy(AW),
            'bias': biasA,
            'learningUpdates': learningUpdates
        })
        
        # Shuffle data for next epoch
        random.shuffle(XBatch)

    return AW, biasA, epochDataList