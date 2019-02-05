class Tree:
    def __init__(self, label="root", treeLevel = 0):
        self.childNodes = {}
        self.treeLevel = treeLevel
        self.label = label
        self.splitOn = ""
        self.entropy = -1
        # Make each possible value an attribute, where the empty object will be released
     
    # When you print a tree, it will print a representation of each node of the tree, its structure, and its attributes (e.g. entropy at that node, etc.)
    # It will print out the root node and all the subtrees, not necessarily in the right order. 
    def __repr__(self):
        treeStr = "Tree Level: "
        count = 0
        while count <= self.treeLevel:
            if (count == 0):
                treeStr += str(count)
            else:
                treeStr += " ------ "
                treeStr += str(count)
            count = count + 1

        print('~~~~~~~~~~Tree Node:~~~~~~~~~~')
        print(treeStr)
        print('treeLevel:', self.treeLevel)
        print("branches:", list(self.childNodes.keys()))
        print("split on attribute:", self.splitOn)
        print("label:", self.label)
        print("entropy at node (before split):", self.entropy)
        print("childNodes", self.childNodes)
        return ""

    def addBranch(self, branch):
        self.childNodes[branch] = None

    def addChildNode(self, branch, node):
        self.childNodes[branch] = node