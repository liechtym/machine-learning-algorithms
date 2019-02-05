import sys
import numpy as np
import utility_functions as uf
import pdb
sys.path.append('data')
import tree

class ID3_boolean:
    def __init__(self):
        self.possible_vals = [-1, 1]
        self.tree = None

    def train(self, XBatch, num, label = "root", index = 0, depth = 50, queue=None):
        self.tree = self.ID3(XBatch, num, label = label, index = index, depth = depth)
        if queue:
            queue.put(self)
        

    def predict(self, instances):
        return self.predict_from_tree(self.tree, instances)

    def predict_from_tree(self, tree, instances):
        if len(tree.childNodes) > 0:
            if tree.splitOn in instances:
                return self.predict_from_tree(tree.childNodes['1'], instances)
            else:
                return self.predict_from_tree(tree.childNodes['-1'], instances)
        else:
            return tree.label

    def ID3(self, XBatch, num, label = "root", index = 0, depth = 50):
        if uf.has_same_label(XBatch):
            label = XBatch[0]['label']
            return tree.Tree(label = label, treeLevel = index)
        elif depth == index:
            label = uf.get_most_common_label(XBatch)
            return tree.Tree(label = label, treeLevel = index)
        else:
            # Create Root node
            rootNode = tree.Tree(label = label, treeLevel = index)
            bestAttr = uf.get_best_attribute(XBatch, num)
            rootNode.splitOn = bestAttr
            for val in self.possible_vals:
                rootNode.addBranch(str(val))
                subset = uf.get_data_subset(val, bestAttr, XBatch)
                if len(subset) == 0:
                    rootNode.addChildNode(str(val), tree.Tree(label = val, treeLevel = index + 1))
                else:
                    rootNode.addChildNode(str(val), self.ID3(subset, num, label = val, index = index + 1, depth = depth))

        return rootNode