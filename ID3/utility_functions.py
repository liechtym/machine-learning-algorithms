import numpy as np
import pdb

def convert_sparse_vector(sparseVector, num):
    outputVector = np.zeros((num,1), dtype='f4')
    for index, value in sparseVector.items():
        index, value = int(index) - 1, int(value)
        outputVector[index][0] = value
    return outputVector

def get_entropy(prob_positive, prob_negative):
    return - np.nan_to_num(prob_negative * (np.log2(prob_negative))) - np.nan_to_num(prob_positive * (np.log2(prob_positive)))

def get_initial_entropy(XBatch, num):
    negative_labels = 0
    positive_labels = 0
    for XObj in XBatch:
        if XObj['label'] == 1:
            positive_labels += 1
        else:
            negative_labels += 1
    prob_negative = negative_labels/len(XBatch)
    prob_positive = positive_labels/len(XBatch)
    return get_entropy(prob_positive, prob_negative)


def get_best_attribute(XBatch, num):
    initial_entropy = get_initial_entropy(XBatch, num)
    attribute_entropy = []
    for index in range(1, num + 1):
        index = str(index)
        # val_dist = {'feature': index,'-1': 0, '1': 0}
        negative_labels = 0
        positive_labels = 0
        for XObj in XBatch:
            if index in XObj['instances']:
                positive_labels += 1
            else:
                negative_labels += 1
        prob_negative = negative_labels/len(XBatch)
        prob_positive = positive_labels/len(XBatch)
        entropy = get_entropy(prob_positive, prob_negative)
        entropy_sum = (prob_negative/len(XBatch)) * entropy + (prob_positive/len(XBatch)) * entropy
        attribute_entropy.append(entropy_sum)
    greatest_entropy = max(attribute_entropy)
    least_entropy = min(attribute_entropy)
    ge_index = attribute_entropy.index(greatest_entropy)
    return str(ge_index + 1)

def get_data_subset(label, attr, XBatch):
    subset = []
    for XObj in XBatch:
        if label == -1 and attr not in XObj['instances']:
            subset.append(XObj)
        elif label == 1 and attr in XObj['instances']:
            subset.append(XObj)
    return subset

def has_same_label(XBatch):
    label = None
    for XObj in XBatch:
        if label and label != XObj['label']:
            return False
        label = XObj['label']
    return True

def get_most_common_label(XBatch):
    positiveLabels = 0
    negativeLabels = 0
    for XObj in XBatch:
        if XObj['label'] == 1:
            positiveLabels += 1
        else:
            negativeLabels += 1
    if positiveLabels > negativeLabels:
        return 1
    else:
        return -1
