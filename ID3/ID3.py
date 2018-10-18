import sys
import numpy as np
import tree
import functions

def ID3(S, Attributes, Label = "root", index = 0, hyperparameter = None):
    # If all examples have the same label or if hyperparameter has been reached: Return a single node tree with the common label
    if (np.unique(S.raw_data[:, 0]).size == 1) or (hyperparameter and hyperparameter == index):
        commonLabel = functions.get_common_value(S.raw_data[:, 0])
        bottomLeafNode = tree.Tree(label = commonLabel, treeLevel = index)
        if (hyperparameter and hyperparameter == index):
            A = functions.get_best_attribute_to_split_on(S, Attributes, 'e')
            bottomLeafNode.entropy = A["entropy"] + A["informationGain"]
        else:
            bottomLeafNode.entropy = 0
        bottomLeafNode.splitOn = None
        return bottomLeafNode
    else:
        # Create Root node
        rootNode = tree.Tree(label = Label, treeLevel = index)
        # Determine attribute to split on
        A = functions.get_best_attribute_to_split_on(S, Attributes, 'e')
        # Assign entropy for current node
        rootNode.entropy = A["entropy"] + A["informationGain"]
        # If the hyperparameter is one above the index, return the rootNode, stop growing the tree
        rootNode.splitOn = A["name"]
        # for each possible value v of that A can take:
        for v in S.get_attribute_possible_vals(A["name"]):
            # Add new branch corresponding to A=v
            rootNode.addBranch(v)
            # Let Sv be the subset of examples in S with A=V
            Sv = S.get_row_subset(A["name"], v)
            # If Sv is empty: add leaf node with common value of Label in S
            if Sv.raw_data.size == 0:
                bottomLeafNode = tree.Tree(label= S.raw_data[0, 0], treeLevel = index)
                bottomLeafNode.entropy = 0
                bottomLeafNode.splitOn = None
                rootNode.addChildNode(v, bottomLeafNode)
            # else below this branch, add the subtree ID3(Sv, Attributes - {A}, Label)
            else:
                attrs = dict(Attributes)
                del attrs[A["name"]]
                rootNode.addChildNode(v, ID3(Sv, attrs, Label = A["name"], index = index + 1, hyperparameter = hyperparameter))
        return rootNode