import csv
import sys
import os
import math
import random
import re
import numpy as np


# Reading Input File and finding the delimiter
input_file=sys.argv[1]
output_file=sys.argv[2]
infile = open(input_file)
file = infile.readlines()
data = list()
dataset = list()

for row in file:
    x=re.split(r'[ ,|;"]+', row)
    if(x[0]==''):
        data.append(x[1:])
    else:
        data.append(x)

#Removing Missing Values

for row in data:
    count =0
    for r in row:
        if ((not r)or(r=='?')):
            count+=1
        else:
            count+=0
    if (count==0):
        dataset.append(row)

#Standarization  and categorical values

for r in range(len(dataset[0])):
    count=0
    sum=0.0
    temp = list()
    for row in range(len(dataset)):
        try:
            num_dataset = float(dataset[row][r])
            count += 1
            sum += num_dataset
            temp.append(num_dataset)
        except ValueError:
            count += 0
            pass
    if(count==(len(dataset))):
        temp = np.array(temp)
        mean=np.mean(temp)
        sd = np.std(temp)
        for row in range(len(dataset)):
            dataset[row][r]=(float(dataset[row][r])-mean)/sd
    else:
        for row in range(len(dataset)):
            temp.append(dataset[row][r])
        temp = np.array(temp)
        uni = np.unique(temp)
        uni=uni.tolist()
        for row in range(len(dataset)):
            dataset[row][r] = uni.index(dataset[row][r])
classSet=list()
for i in range(len(dataset)):
    classSet.append(dataset[i][-1])
classSet = np.array(classSet)
max = np.amax(classSet)
min = np.amin(classSet)
classSet = classSet.tolist()
for i in range(len(dataset)):
    dataset[i][-1] = (float(dataset[i][-1]-min)/(max-min))

#writing the processed set to file
with open(output_file, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(dataset)




