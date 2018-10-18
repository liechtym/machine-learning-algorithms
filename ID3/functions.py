import numpy as np

def get_label_count (dataSubset, positiveLabel):
    numberNegative = 0
    numberPositive = 0
    for label in dataSubset[:, 0]:
        if label == positiveLabel:
            numberPositive += 1
        else:
            numberNegative += 1
    return numberNegative, numberPositive

def calc_entropy (positiveData, negativeData):
    fractionPositive = positiveData/(positiveData + negativeData)
    fractionNegative = 1 - fractionPositive
    return np.nan_to_num(- fractionPositive * np.log2(fractionPositive) - fractionNegative * np.log2(fractionNegative))

# Returns a dictionary, with a key for each attribute, and a dictionary for each value. The dictionary for each key gives the entropy and the information gain of that attribute.
def calc_impurity_by_attribute(trainingData, attributes, positiveLabel):
    dataLength = np.size(trainingData.raw_data, 0)

    numNegative, numberPositive = get_label_count(trainingData.raw_data, positiveLabel)
    entropyBeforeSplitting = calc_entropy(numNegative, numberPositive)

    attributeResults = {}
    # Iterate through attributes
    for attr in attributes:
        # For each attribute, iterate through the possible values
        attrCumulativeEntropy = 0
        attributeValues = trainingData.get_attribute_possible_vals(attr)
        for attrVal in attributeValues:
            # Create a subset from the attribute values that isolate a certain value
            attributeValSubset = trainingData.get_row_subset(attr, attrVal)
            # Assign each label to a total
            if np.size(attributeValSubset.raw_data, 0) > 0:
                numberNegative, numberPositive = get_label_count(attributeValSubset.raw_data, positiveLabel)
                labelVals = attributeValSubset.raw_data[:, 0]
                valueEntropy = calc_entropy(numberPositive, numberNegative)
                attrCumulativeEntropy += (valueEntropy) * (labelVals.size/dataLength)

        attributeResults[attr] = {
            "name": attr,
            "index": trainingData.get_column_index(attr),
            "entropy": attrCumulativeEntropy,
            "informationGain": entropyBeforeSplitting - attrCumulativeEntropy,
            "attributeValues": attributeValues
        }
    return attributeResults

def get_attribute_with_highest_gain(attributeDict):
    attrWithHighestGainSoFar = {}
    for attr in attributeDict:
        if not bool(attrWithHighestGainSoFar) or (attributeDict[attr]['informationGain'] > attrWithHighestGainSoFar['informationGain']):
            attrWithHighestGainSoFar = attributeDict[attr]
    return attrWithHighestGainSoFar

def get_best_attribute_to_split_on(trainingData, attributes, positiveLabel):
    impurityCalcs = calc_impurity_by_attribute(trainingData, attributes, positiveLabel)
    attributeWithHighestGain = get_attribute_with_highest_gain(impurityCalcs)
    return attributeWithHighestGain

def get_common_value(column):
    uniqueValues, valueCounts = np.unique(column, return_counts=True)
    indexOfLargestCount = np.argmax(valueCounts)
    mostCommonValue = uniqueValues[indexOfLargestCount]
    return mostCommonValue