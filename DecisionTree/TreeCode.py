class TreeNode():
    def __init__(self, nodeValue):
        self.value = nodeValue
        self.left = None
        self.right = None

    def getValue(self):
        return self.value


class Tree():
    def __init__(self, node):
        self.root = node
        self.leftChild = None
        self.rightChild = None

    def getRootValue(self):
        return self.root.getValue()

    def setRight(self, newNode):
        self.rightChild = newNode

    def setLeft(self, newNode):
        self.leftChild = newNode

    def getLeftChild(self):
        return self.leftChild

    def getRightChild(self):
        return self.rightChild

    def printTree(self):
        if (self == None):
            print('Empty')
        else:
            self.printThisTree(self.root)

    def printThisTree(self, node):
        if (node != None):
            print(self.getRootValue())
            self.printThisTree(self.getLeftChild())
            self.printThisTree(self.getRightChild())


myTree = Tree(TreeNode('Xo'))
myTree.setLeft(TreeNode('XM'))
myTree.setRight(TreeNode('XR'))
myTree.printTree()