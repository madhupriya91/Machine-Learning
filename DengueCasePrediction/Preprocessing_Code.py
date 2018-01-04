import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

output_file=sys.argv[2]  #Output file location

# Load dataset
dataset = pd.read_csv(sys.argv[1])
dataset = dataset.drop(['week_start_date'],axis =1 )  #Dropping the unnecessary feature

#Extracting feature values as x matrix and the labels as y matrix
x = dataset.ix[:,0:23].values
y = dataset.ix[:,23].values

# Keeping year and week number aside so as to not normalising it
second = x[:,1:3]

#Encoding nominal values : city names

le=LabelEncoder()
le.fit(x[:,0])
X_scaled=le.transform(x[:,0]).round(0)

#Replacing missing values in the dataset with the most frequently occuring value for that feature

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

#Standardization
scaler = preprocessing.StandardScaler().fit(imp.fit_transform(x[:,3:23]))
scaler.mean_
scaler.scale_

New_scaled=scaler.transform(imp.fit_transform(x[:,3:23]))

#Final dataset obtaines after joining the encoded city names, unstandardized year and  week number and remaining data after standardising
final=(np.c_[X_scaled,second,New_scaled])

#merge into single matrix
final_dataset = (np.c_[final,y])

#write as file output
pd.DataFrame(final_dataset).to_csv(sys.argv[2], index=False, header=False)