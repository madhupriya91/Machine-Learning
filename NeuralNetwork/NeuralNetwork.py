import sys
import random
import numpy as np
import math


class Neuron():
    def __init__(self,weights,inputs,flag):
        self.inWeights=weights

        self.outY=0
        self.inputX=inputs
        self.inFlag=flag
        self.deltaWeights=[]
        self.delta=0
    def sigmoid(self):
        sum=0
        if self.inFlag==1:
            self.outY=self.inputX[0]
        else:
            for i in range(len(self.inputX)):
                sum+=self.inputX[i]*self.inWeights[i]
                self.outY=1.0/(1.0+np.exp(-sum))
        return self.outY
    def calDelta(self,nextLayer,neuronIndex,outLayFlag,target):
        sumTerm=0
        if(outLayFlag==1):
            self.delta=self.outY*(1-self.outY)*(float(target)-self.outY)
            return self.delta
        for nextNeuron in range(len(nextLayer)):
            sumTerm+=nextLayer[nextNeuron].inWeights[neuronIndex+1]*nextLayer[nextNeuron].delta
            #self.delta=self.outY*(1-self.outY)*sumTerm
        self.delta = self.outY * (1 - self.outY) * sumTerm
        return self.delta
    def calDeltaWeights(self,prevLayer):
        eta=0.5
        self.deltaWeights=[]
        self.deltaWeights.append(eta * self.delta)
        for neuronIndex in range(len(prevLayer)):
            #self.deltaWeights[neuronIndex]=eta*self.delta*prevLayer[neuronIndex].outY
            #self.deltaWeights.append(eta * self.delta)
            self.deltaWeights.append(eta * self.delta * prevLayer[neuronIndex].outY)
            #print(self.deltaWeights)
        #for i in range(len(self.inWeights)):
         #   self.inWeights[i]+=self.deltaWeights[i]
    def updateWeights(self):
        for i in range(len(self.inWeights)):
            self.inWeights[i]+=self.deltaWeights[i]

    def updateInputs(self,inputs):
        self.inputX=inputs




trainPercent=int(sys.argv[2])


hidLayCount=int(sys.argv[4])
neuronCount=[]
for i in range(5,5+hidLayCount):
    neuronCount.append(int(sys.argv[i]))
maxIterations=float(sys.argv[3])
inputs=[]
instances=0
classCount=0
#allWeights=[[-.4 ,0.2,.4,.5],[0.2,-.3,.1,.2],[.1,-.3,-.2]]
file = open(str(sys.argv[1]))

data = [[]]
for line in file:
    line = line.strip("\r\n")
    data.append(line.split(','))
    instances += 1
data.remove([])
classList=[]
inputData=[[]]
#print(data)

#print(data)



inputCount=len(data[0])-1
layers=[[]]
#print(inputCount)
totLayCount=hidLayCount+2

def createNetwork():
    for l in range(totLayCount):
        neurons = []
        allWeightsIndex = 0
        if (l != totLayCount - 1) and l != 0:
            for n in range(neuronCount[l-1]):
                thisWeights=[random.uniform(0,1)]
                #thisWeights = [1]
                thisInputs = [1]
                for i in range(len(layers[l - 1])):
                    #thisInputs.append(layers[l - 1][i].outY)
                    thisWeights.append(random.uniform(0,1))
                    #thisWeights.append(1)
                    # thisWeights=allWeights[allWeightsIndex]
                    # allWeightsIndex+=1
                neurons.append(Neuron(thisWeights, thisInputs, 0))
                #neurons[n].sigmoid()
            layers.append(neurons)
        elif l == 0:
            for i in range(inputCount):
                thisInputs = []
                #thisInputs.append(float(data[0][i]))
                neurons.append(Neuron(0, thisInputs, 1))
                #neurons[i].sigmoid()
            layers[l] = neurons
        else:
            thisWeights = [random.uniform(0,1)]
            #thisWeights = [1]
            # thisWeights = allWeights[allWeightsIndex]
            thisInputs = [1]
            for i in range(len(layers[l - 1])):
                #thisInputs.append(layers[l - 1][i].outY)
                thisWeights.append(random.uniform(0, 1))
                #thisWeights.append(1)
            neurons.append(Neuron(thisWeights, thisInputs, 0))
            #neurons[0].sigmoid()
            layers.append(neurons)



def forwardPass(instance):
    for layerInd in range(len(layers)):
        for neuronInd in range(len(layers[layerInd])):
            if(layerInd!=0):
                thisInputs=[1]
                for i in range(len(layers[layerInd - 1])):
                    thisInputs.append(layers[layerInd - 1][i].outY)
                layers[layerInd][neuronInd].updateInputs(thisInputs)
                layers[layerInd][neuronInd].sigmoid()
            else:
                thisInputs = []
                thisInputs.append(float(instance[neuronInd]))
                layers[layerInd][neuronInd].updateInputs(thisInputs)
                layers[layerInd][neuronInd].sigmoid()

def backwardPass():
    for l in range(totLayCount - 1, -1, -1):
        if l != totLayCount - 1:
            for thisNeuronInd in range(len(layers[l])):
                layers[l][thisNeuronInd].calDelta(layers[l + 1], thisNeuronInd, 0, 0)
                if l != 0:
                    layers[l][thisNeuronInd].calDeltaWeights(layers[l - 1])
        else:
            for thisNeuronInd in range(len(layers[l])):
                layers[l][thisNeuronInd].calDelta(layers[l], thisNeuronInd, 1, data[0][-1])
                layers[l][thisNeuronInd].calDeltaWeights(layers[l - 1])


def updateNetwork():
    for layer in range(1, len(layers)):
        for neuron in range(len(layers[layer])):
            layers[layer][neuron].updateWeights()


def trainNetwork(instance):
    forwardPass(instance)
    backwardPass()
    updateNetwork()

def testNetwork(instance):
    forwardPass(instance)

'''iterations=1000
maxIterations=iterations/len(data)
for itr in range(maxIterations)
    for entry in data:
        trainNetwork(entry)

'''

def printWeights():
    for layerInd in range(len(layers)):
        if layerInd==0 or layerInd==totLayCount-1:
            print("Layer " + str(layerInd)+" :")
        else:
            print("Hidden layer "+str(layerInd))
        for neuronInd in range(len(layers[layerInd])):
            print("\n Neuron "+str(neuronInd)+" weights "+ str(layers[layerInd][neuronInd].inWeights))



createNetwork()
#print("Done")
printWeights()
squareError=0
iterations=0
random.shuffle(data)
trainLimit=int((len(data)*trainPercent)/100)
entryInd=0
#####################
while iterations<maxIterations:
    for entry in data:
        trainNetwork(entry)
        error = float(data[entryInd][-1]) - layers[totLayCount - 1][len(layers[totLayCount - 1]) - 1].outY
        squareError += math.pow(error, 2)
    iterations+=1
    trainError=squareError/len(data)
    if trainError==0:
        print("Converged !")
        break

print("Training error : "+str(trainError))



##################################
#  TESTING

for entryInd in range(trainLimit,len(data)):
    testNetwork(data[entryInd])
    error=float(data[entryInd][-1])-layers[totLayCount-1][len(layers[totLayCount-1])-1].outY
    squareError+=math.pow(error,2)


print("Test error : "+str(squareError/len(data)))
