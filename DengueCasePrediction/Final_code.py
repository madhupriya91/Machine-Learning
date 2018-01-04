'''
CS6375.004: Machine Learning
Project: DengAI: Predicting Disease Spread

Team Members : Nandish Muniswamappa (nxm160630)
               Bhargav Lenka (bxl171030)
               Madhupriya Pal (mxp162030)
               Masoud Shahshahani (mxs161831)

File name: Final_code.py            
Input argument format : <Training Data set> <Test Data set> <Output file> 
               
'''
# Importing all required libraries
import sys
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import median_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor


# Ignore display of unnecessary warnings on console
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

dataset = pd.read_csv(sys.argv[1], header = None)
testdataset = pd.read_csv(sys.argv[2], header = None)

names = ['city','year',	'weekofyear','ndvi_ne',	'ndvi_nw',	'ndvi_se',	'ndvi_sw',	'precipitation_amt_mm',	'reanalysis_air_temp_k',	'reanalysis_avg_temp_k',	'reanalysis_dew_point_temp_k',	'reanalysis_max_air_temp_k',	'reanalysis_min_air_temp_k',	'reanalysis_precip_amt_kg_per_m2',	'reanalysis_relative_humidity_percent',	'reanalysis_sat_precip_amt_mm',	'reanalysis_specific_humidity_g_per_kg',	'reanalysis_tdtr_k',	'station_avg_temp_c',	'station_diur_temp_rng_c',	'station_max_temp_c',	'station_min_temp_c',	'station_precip_mm'
]
#training data and validation data (20% of training data)
X = np.array(dataset.ix[:,0:22].values)
y = np.array(dataset.ix[:,23].values)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#test data
Xtest = np.array(testdataset.ix[:,0:22].values)

'''
#grid search to find best parameters
param_grid = {
                 'n_estimators': [100,200,250,300,350],
                 'learning_rate': [.09,.08,.1,.2],
                'loss':['huber','quantile'],
                'max_depth': [3,5,7,9],
             }

gb = GradientBoostingRegressor()
grid_gbr = GridSearchCV(gb, param_grid, cv=10)
grid_gbr.fit(X_train, y_train)
grid_gbr.cv_results_
print(grid_gbr.score(X_test, y_test))
print("Best Parameters:")
print(grid_gbr. best_params_)
'''
print('\nDengAI: Predicting Disease Spread')
print('__________________________________')

# fitting our model
gbr = GradientBoostingRegressor(n_estimators=320,learning_rate =0.09, loss='huber',
                                min_samples_split= 4, max_depth=8,  
                                max_features=16)
gbr = gbr.fit(X_train, y_train)


# Evaluation Metrics
print("Gradient Boosting : Evaluation Metrics")
accuracy = gbr.score(X_test, y_test)
print("Accuracy:", round(accuracy*100,4), '%')
mae = median_absolute_error(y_test, gbr.predict(X_test))
print("Median Absolute Error:", round(mae,4))
meanae = mean_absolute_error(y_test, gbr.predict(X_test))
print("Mean Absolute Error:", round(meanae,4))

print('\nOther Machine Learning Techniques')
rfr = RandomForestRegressor(n_estimators=303, min_samples_split=4, max_depth =9,
                            max_features=12, bootstrap=False)
rfr = rfr.fit(X_train, y_train)
print("Random Forest Accuracy:", round((rfr.score(X_test, y_test))*100,4), '%')


knr = KNeighborsRegressor(algorithm='ball_tree', n_neighbors=5, leaf_size=32,
                          weights='distance', p=1)
knr = knr.fit(X_train, y_train)
print("K-Nearest Neighbors Accuracy:", round((knr.score(X_test, y_test))*100,4), '%')
  
#predictions on test set
  
print('\nPlot of Relative importance of attributes in data set ')
print('______________________________________________________')

predictions=gbr.predict(Xtest)
predInt=[]
for prediction in predictions:
    if prediction<0:
        prediction=0
    predInt.append(int(prediction))

#Plot feature importance
feature_importance = gbr.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance, align='center')
plt.yticks(pos, names)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

#writing output to a csv file
year=[]
week=[]
city=[]
for row in range(len(Xtest)):
    year.append(int(Xtest[row,1]))
    week.append(int(Xtest[row, 2]))
    cityCode=int(Xtest[row, 0])
    if cityCode==0:
        city.append("iq")
    else:
        city.append("sj")

cityNP=np.array(city)
yearNP=np.array(year)
weekNP=np.array(week)
predIntNP=np.array(predInt)
output = np.c_[cityNP,yearNP,weekNP,predIntNP]
names2 = ['city','year','week','prediction']
pd.DataFrame(output).to_csv(sys.argv[3],header=names2,index = False)