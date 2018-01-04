import math
import random
#import numpy as np
import sys
#import csv
#import pandas as pd
#from pandas import Series,DataFrame
import numpy as np
import TreeNode
#print(dataSet)
instances=0


file = open(str(sys.argv[1]))

data = [[]]
for line in file:
    line=line.strip("\r\n")
    data.append(line.split(','))
    instances+=1
data.remove([])
attributes = data[0]

#print(data)
colIndexClass=attributes.index('Class')
#print(colIndexClass)
classf=[]
for entry in data:
    #classf.append(entry[3])
    classf.append(entry[colIndexClass])
    del entry[colIndexClass]
#print(data)
dataSet={}
instances-=1
#print('  OUTPUT')
del data[0]
#print(data)
attrCount=len(attributes)
for attr in range(len(attributes)):
    #print(attr)
    attrVal=[]
    for entry in data:
        attrVal.append(entry[attr])
        dataSet[attributes[attr]]=attrVal
del classf[0]
tup=attributes
#print(classf)
#dataSet_df = pd.read_csv(str(sys.argv[1]))
#attributes=dataSet_df(userow=[0])
#print(attributes)
#classf = pd.read_csv(str(sys.argv[1]),usecols=[3])
#dataSet = pd.read_csv(str(sys.argv[1]),usecols=[0,1,2])
#print(classf)
#print(dataSet)
#file = open(str(sys.argv[1]))
"""
IMPORTANT: Change this variable too change target attribute 
"""
#target = "Class"
#target.remove(0)
#extract training data
#data = [[]]
#for line in file:
#    line=line.strip("\r\n")
#    data.append(line.split(','))
#data.remove([])
#attributes = data[0]
#print(data)

#classf= [record[attributes.index(target)] for record in data]
#classf.remove('Class')
#data.remove('Class')
#print(classf)
#data.remove(attributes)
#print(data)
#classf= [record[attributes.index(target)] for record in data]
#attributes.remove('Class')
#attributes.remove('Class')
#print(attributes)
#classf=
#print(data)
#print(classf)
#print(attributes)
#print(classf)
#print(data)

#import xlrd
##orksheet = workbook.sheet_by_index(0)
#classf=sheet.col(3)
#file = open(str(sys.argv[1]))
"""
IMPORTANT: Change this variable too change target attribute 
"""
#target = "Class"
#extract training data
#data = [[]]
#for line in file:
##	data.append(line.split(','))
#data.remove([])
#attributes = data[0]
#data.remove(attributes)
#print(attributes)
#print(data)
#classf=data[:]
#print(classf)
spaceC=-1
#dataSet = {'X1': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1], 'X2': [0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
         #  'X3': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]}
entropyHash={}
nodesCount=[0]
leafCount=[0]
#classf = [1,1,0,0,0,1,0,1,0,1]
#tup=['X1','X2','X3']
#entropyAll={'X1':0,'X2':0,'X3':0}
size = len(dataSet)
dataIndices = []
for i in range (len(classf)):
    dataIndices.append(i)
    parAttr="none"
dataVal = dataSet.values()

def match(X,branch,classVal,indices):
    count = 0
    i = 0
    if (branch == "left"):
        attrV = '0';
    else:
        attrV = '1'
    if (classVal == "N"):
        classV = '0'
    else:
        classV = '1'
    vals=dataSet.get(X)
    for i in indices:
        #print(vals[i])
        #print(classf[i])
        if (vals[i] == attrV and classf[i] == classV):
            count += 1

    return count


#try:
 #   z = x / y
#except ZeroDivisionError:
 #   z = 0
def calEntropy(N,Y):
    YN = Y + N
    if YN == 0:
        return 0
    else:
     pY = Y / (Y + N)
     pN = N / (Y + N)
     if (pY != 0 and pY != 1):
        lgY = math.log(pY, 2)
     else:
        lgY = 0
     if (pN != 0 and pY != 1):
        lgN = math.log(pN, 2)
     else:
        lgN = 0

     H = -((pY) * lgY + (pN) * lgN)
     return H

'''def calEntropy(N,Y):
    YN = Y+N
    pY = Y / (Y + N)
    pN = N / (Y + N)
    if (pY != 0 and pY != 1):
        lgY = math.log(pY, 2)
    else:
        lgY = 0
    if (pN != 0 and pY != 1):
        lgN = math.log(pN, 2)
    else:
        lgN = 0

    H = -((pY) * lgY + (pN) * lgN)
    return H
'''
def childEntropy(branch):
    childH = calEntropy(branch[0],branch[1])
    return childH

def avgEnt(attr,indices):
    leftMatch = []
    rightMatch = []
    leftMatch.append(match(attr, "left", "N",indices))
    leftMatch.append(match(attr, "left", "Y",indices))
    rightMatch.append(match(attr, "right", "N",indices))
    rightMatch.append(match(attr, "right", "Y",indices))
    H1 = childEntropy(leftMatch)
    H2 = childEntropy(rightMatch)
    avgH = ((leftMatch[0] + leftMatch[1])/len(indices)) * H1 + ((rightMatch[0] + rightMatch[1])/len(indices)) * H2
    return(avgH)

def entropy(attr,indices):
    yes = 0
    no = 0
    if (attr == 'none'):
       for value in classf:
           if (value=='0'):
               yes += 1
           else:
               no += 1
       return(calEntropy(no,yes))
    return(avgEnt(attr,indices))

    #elif(attr in entropyHash):
     #   return entropyHash.value(attr)
    #leftMatch = []
    #rightMatch = []
    #leftMatch.append(match(attr, "left", "N",indices))
    #leftMatch.append(match(attr, "left", "Y",indices))
    #rightMatch.append(match(attr, "right", "N",indices))
    #rightMatch.append(match(attr, "right", "Y",indices))
    #H1 = childEntropy(leftMatch)
    #H2 = childEntropy(rightMatch)
    #avgH = ((leftMatch[0] + leftMatch[1])/len(indices)) * H1 + ((rightMatch[0] + rightMatch[1])/len(indices)) * H2
    #avgH=(leftMatch[0]/(leftMatch[0]+rightMatch[1]))*H1 + (rightMatch[1]/(leftMatch[0]+rightMatch[1]))*H2
    # entropyHash[attr]=avgH
    return(avgH)

def iGain(X, parent, indices):
    if (parent in entropyHash):
        parentH=entropyHash[parent]
        #parentH=entropyHash.value(parent)
    else:
        parentH = entropy(parent,indices)
    thisH = entropy(X,indices)
    infGain = parentH - thisH
    return (infGain)


def bestAttr(attrs, parAttr, indices, dataSet):
    ig = []
    for attr in attrs:
        ig.append(iGain(attr, parAttr, indices))
    IG = max(ig)
    index = ig.index(IG)
    #return index

    return (attrs[index])

def makeTree(dataSet,attrs,indices,classf,parAttr,spaceC):
    allEqual = 1
    j=0
    #for i in indices:
    #s=''
    for i in range(len(indices)-1):
        if(classf[indices[i]]!=classf[indices[i+1]]):
            allEqual=0
            break
        else:
            j=i
            continue
    if allEqual==1:
        #s=classf[indices[j]]
       # return(s.lstrip())
        leafCount[0]+=1
        if(len(indices)!=0):
            return(classf[indices[j]])
        else:
            return(classf[0])
        #print(classf[indices[i]])
        #exit()

    if(attrs==[]):
        counts = np.bincount(classf)
        return np.argmax(counts)
        #exit()




    parentAttr = bestAttr(attrs,parAttr,indices,dataSet)
    entropyHash[parentAttr] = avgEnt(parentAttr,indices)
    nodesCount[0]+=1
    tree = {parentAttr: {}}
    #if (parentAttr != 'None'):
     #   print(parentAttr)
    #j=0
    newAttrs=[]
    for i in range(len(attrs)):
        if(attrs[i]!=parentAttr):
            newAttrs.append(attrs[i])
            #j+=1
    #if(parentAttr!='None'):
     #   print(parentAttr)
    #newAttrs=attrs
    #del newAttrs[ParentAttrIndex]

    spaceC+=1
   # newAttrs.remove(parentAttrIndex)
    #newIndices=[]
    attrBranches=['0','1']
    attrValues=dataSet.get(parentAttr)
    for branch in attrBranches:
        newIndices = []

        #print(parentAttr+'='+branch+':')
        for i in indices:
            if(attrValues[i]==branch):
                newIndices.append(i)
        #print('\n'+spaceC*'| '+parentAttr + '=' + str(branch)+':',end='')
       # print(str(makeTree(dataSet, newAttrs, newIndices, classf, parentAttr, spaceC))+spaceC * '| ' + parentAttr + '=' + str(branch) + ':')
        #print(str(makeTree(dataSet, newAttrs, newIndices, classf, parentAttr,spaceC,leafCount)),end='')
        tree[parentAttr][branch]=makeTree(dataSet, newAttrs, newIndices, classf, parentAttr,spaceC)
        #print(tree)
    return tree

        #result=makeTree(dataSet, newAttrs, newIndices, classf, parentAttr)
        #if(result!='None'):
        #print (result)
        #print(parentAttr + '=' + str(branch) + ':'+str(result))
       # for attrV in attrValues:
        #    if(attrV==branch):
         #       newIndices.append(attrValues.index(attrV))

def printTree(tree,height):
    attrName=list(tree.keys())[0]
    #print(attrName)
    attrTreeValue=tree[attrName]
    branches=list(attrTreeValue.keys())
    #print(branches)
    for branch in branches:
        value=attrTreeValue[branch]
        if isinstance(value, dict):
            modelString = (height * '| ') + attrName + ' = ' + str(branch) + ' : '
            print(modelString)
            printTree(value, height + 1)
        else:
            modelString = (height * '| ') + attrName + ' = ' + str(branch) + ' : ' + str(value)
            print((height * '| ') + attrName + ' = ' + str(branch) + ' : ' + str(value) )
#print"training data file: " + str(sys.argv[1])

tree=makeTree(dataSet,tup,dataIndices,classf,parAttr,spaceC)
print(tree)
printTree(tree,0)
'''pruneTree(tree,0)
#label=0
def pruneTree(tree,label):
    tempDict = tree.copy()
    while (isinstance(tempDict, dict)):
        root = TreeNode.TreeNode(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])
        oldDict = tempDict
        tempDict = tempDict[list(tempDict.keys())[0]]
        if(tempDict() = random):
            remove.tempDict(random) 
            
        index = attributes.index(root.attrName)
        value = entry[index]

    child = TreeNode.TreeNode(value, tempDict[value])
    result = tempDict[value]
    tempDict = tempDict[value]
    # break

'''



print('\nInstances:'+str(instances))
print('No of attr:'+str(attrCount))
print('No of leaves '+str(leafCount[0]))
print('No of nodes '+str(nodesCount[0]))
#tempDict = list(tree.copy())[0]
instances=0
nodeCount=[0]
leafCount=[0]
attrCount=0
attributes=[]

testData = [[]]
print("test data file: " + str(sys.argv[2]))
f = open(str(sys.argv[2]))
# extract test data
for line in f:
    line = line.strip("\r\n")
    testData.append(line.split(','))
testData.remove([])
attributes=testData[0]
attrCount=len(attributes)-1
count = 0
correctPred = 0
instances = 0
total=0
hits=0
label=0
defaultTag=0
for entry in testData:
    #instances += 1
    if str(entry[len(entry) - 1]) != 'Class':
        instances += 1
        tempDict =tree.copy()
        result = ""
        while (isinstance(tempDict, dict)):
            #label += 1
            root = TreeNode.TreeNode(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])
            oldDict = tempDict
            tempDict = tempDict[list(tempDict.keys())[0]]
            index = attributes.index(root.attrName)
            value = entry[index]
            #if str(index) == str(value):
            #if str(value) == str(entry[len(entry) - 1]):
              #  correctPred += 1
             #   break
            #label+=1
            child = TreeNode.TreeNode(value, tempDict[value])
            result = tempDict[value]
            tempDict = tempDict[value]
            #break
        if(result==entry[len(entry)-1]):
            correctPred+=1


if __name__ == '__makeTree__':
    makeTree()

print("No. of correct Predictions: " + str(correctPred))
print("Total no. of instances : " + str(instances))
accuracy = float(correctPred) / instances * 100
print(str(accuracy) + '%' + ' accuracy on test set')

#print('HASH:'+str(entropyHash))
#print(tempDict.keys()[0])
#print(tempDict[tempDict.keys()[0]])
#rootNode=TreeNode.TreeNode(tree.)
print('\nInstances:'+str(instances))
print('No of attr:'+str(attrCount))
print('No of leaves '+str(leafCount[0]))
print('No of nodes '+str(nodeCount[0]))
label=0


class Node():
    def __init__(self,attrV,leftC,rightC,parentV):
        self.attr=attrV
        self.parent=parentV
        self.left=leftC
        self.right=rightC




def computeTree(root,tree):
    if isinstance(tree, dict):
        tempDict = tree.copy()
    result = ""
    if root is None:
        root = Node(list(tree.keys())[0], None, None, None)
    elif (root.attr=='0' or root.attr=='1'):
        return root.parent
    tempNode=root
    while (isinstance(tempDict, dict)):
        # label += 1
        parentName=tempNode.attr
        tempDict = tempDict[list(tempDict.keys())[0]]
        #tempNode.parent=parentName
        leftDict=tempDict[list(tempDict.keys())[0]]
        rightDict=tempDict[list(tempDict.keys())[1]]
        if (leftDict == '0' or leftDict == '1' ):
            tempNode.left = Node(leftDict, None, None, tempNode)
        else:
            tempNode.left = Node(list(leftDict.keys())[0], None, None, tempNode)
        if (rightDict == '0' or rightDict == '1' ):
            tempNode.right = Node(list(rightDict.keys())[0],None,None,tempNode)
        else:
            tempNode.right = Node(rightDict, None, None, tempNode)
        #root=tempNode
        #tempNode=tempNode.left
        #tempDict=tempDict[list(tempDict.keys())[0]]
        computeTree(tempNode.left,tempDict[list(tempDict.keys())[0]])
        #tempDict = tempDict[list(tempDict.keys())[0]]
        #tempNode = tempNode.right
        computeTree(tempNode.right, tempDict[list(tempDict.keys())[1]])
        #tempDict = tempDict[list(tempDict.keys())[0]]

        return root

root=Node('X1',None,None,None)
computeTree(root,tree)







