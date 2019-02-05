import pdb
import numpy as np

class simple_naive_bayes:
    def __init__(self):
        self.positive_counts = {}
        self.negative_counts = {}
        self.positive_labels = 0
        self.negative_labels = 0
        self.example_count = 0
        # This is a boolean classifier
        self.labels = [-1, 1]
        
    def train(self, XBatch, num, smooth_by=1):
        self.num = num
        self.smooth_by = smooth_by
        self.example_count = len(XBatch)
        for XObj in XBatch:
            if XObj['label'] == 1:
                self.positive_labels += 1
                for instance in XObj['instances']:
                    if instance not in self.positive_counts:
                        self.positive_counts[instance] = 1
                    else:
                        self.positive_counts[instance] += 1
            else:
                self.negative_labels += 1
                for instance in XObj['instances']:
                    if instance not in self.negative_counts:
                        self.negative_counts[instance] = 1
                    else:
                        self.negative_counts[instance] += 1
    
    def predict(self, instances):
        # First, calculate the value for the negative label
        negativePrior = np.log10(self.negative_labels/self.example_count)
        negativeLikelihoodSum = 0
        for instanceIndex in range(1, self.num + 1):
            instanceIndex = str(instanceIndex)
            instanceCount = self.negative_labels
            if instanceIndex in self.negative_counts:
                # If this feature has positive values
                if instanceIndex in instances:
                    instanceCount = self.negative_counts[instanceIndex]
                else:
                    instanceCount = self.negative_labels - self.negative_counts[instanceIndex]
            likelihood = np.log10((instanceCount + self.smooth_by)/(self.negative_labels + (2 * self.smooth_by)))
            negativeLikelihoodSum += likelihood
        negativeLikelihood = negativeLikelihoodSum/self.num
        negativeLabelVal = negativePrior + negativeLikelihoodSum

        # Next, calculate the value for the positive label
        positivePrior = np.log10(self.positive_labels/self.example_count)
        positiveLikelihoodSum = 0
        for instanceIndex in range(1, self.num + 1):
            instanceIndex = str(instanceIndex)
            instanceCount = self.positive_labels
            if instanceIndex in self.positive_counts:
                # If this feature has positive values
                if instanceIndex in instances:
                    instanceCount = self.positive_counts[instanceIndex]
                else:
                    instanceCount = self.positive_labels - self.positive_counts[instanceIndex]
            likelihood = np.log10((instanceCount + self.smooth_by)/(self.positive_labels + (2 * self.smooth_by)))
            positiveLikelihoodSum += likelihood
        positiveLabelVal = positivePrior + positiveLikelihoodSum

        # Make prediction based on values
        return -1 if max(negativeLabelVal, positiveLabelVal) == negativeLabelVal else 1