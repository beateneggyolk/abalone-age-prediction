import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew
%matplotlib inline
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import  train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.linear_model import  Ridge
from sklearn.svm import SVR

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
print(os.listdir("C:/Users/yeswa/Desktop/semester 6/Machine Learning/mldatasets"))
data=pd.read_csv("C:/Users/yeswa/Desktop/semester6/MachineLearning/mldatasets/abalone.csv",delimiter=',',encoding="utf-8-sig")
data['age']=data['rings']+1.5
data.drop('rings',axis=1,inplace=True)
print('This dataset has {} observations with {} features.'.format(data.shape[0], data.shape[1]))
data.columns

data.hist(figsize=(20,10), grid=False, layout=(2, 4), bins = 30)
numerical_features = data.select_dtypes(include=[np.number]).columns
categorical_features = data.select_dtypes(include=[np.object]).columns
numerical_features
categorical_features

skew_values = skew(data[numerical_features], nan_policy = 'omit')

dummy = pd.concat([pd.DataFrame(list(numerical_features), columns=['Features']), 
           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1)
dummy.sort_values(by = 'Skewness degree' , ascending = False)
missing_values = data.isnull().sum().sort_values(ascending = False)

percentage_missing_values = (missing_values/len(data))*100
pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])
sns.countplot(x = 'sex', data = data, palette="Set3")
plt.figure(figsize = (20,7))
sns.swarmplot(x = 'sex', y = 'age', data = data, hue = 'sex')
sns.violinplot(x = 'sex', y = 'age', data = data)

data.groupby('sex')[['length', 'diameter', 'height', 'whole weight', 'shucked weight',
       'viscera weight', 'shell weight', 'age']].mean().sort_values('age')
sns.pairplot(data[numerical_features])

plt.figure(figsize=(20,7))
sns.heatmap(data[numerical_features].corr(), annot=True)
data = pd.get_dummies(data)
dummy_data = data
data.boxplot( rot = 90, figsize=(20,5))

var = 'viscera weight'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)
data.drop(data[(data['viscera weight']> 0.5) & (data['age'] < 20)].index, inplace=True)
data.drop(data[(data['viscera weight']<0.5) & (data['age'] > 25)].index, inplace=True)

var = 'shell weight'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)
data.drop(data[(data['shell weight']> 0.6) & (data['age'] < 25)].index, inplace=True)
data.drop(data[(data['shell weight']<0.8) & (data['age'] > 25)].index, inplace=True)

var = 'shucked weight'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)
data.drop(data[(data['whole weight']>= 2.5) & (data['age'] < 25)].index, inplace=True)
data.drop(data[(data['whole weight']<2.5) & (data['age'] > 25)].index, inplace=True)

var = 'diameter'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True) data.drop(data[(data['diameter']<0.1) & (data['age'] < 5)].index, inplace=True)
data.drop(data[(data['diameter']<0.6) & (data['age'] > 25)].index, inplace=True)
data.drop(data[(data['diameter']>=0.6) & (data['age']< 25)].index, inplace=True)

var = 'height'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)
data.drop(data[(data['height']>0.4) & (data['age'] < 15)].index, inplace=True)
data.drop(data[(data['height']<0.4) & (data['age'] > 25)].index, inplace=True)

var = 'length'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)
data.drop(data[(data['length']<0.1) & (data['age'] < 5)].index, inplace=True)
data.drop(data[(data['length']<0.8) & (data['age'] > 25)].index, inplace=True)
data.drop(data[(data['length']>=0.8) & (data['age']< 25)].index, inplace=True)

X = data.drop('age', axis = 1)
y = data['age']

standardScale = StandardScaler()
standardScale.fit_transform(X)
selectkBest = SelectKBest()

X_new = selectkBest.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.25)
np.random.seed(10)

def rmse_cv(model, X_train, y):
    rmse =- (cross_val_score(model, X_train, y, scoring='neg_mean_squared_error', cv=5))
    return(rmse*100)
    models = [LinearRegression(),
             Ridge(),
             SVR(),
             RandomForestRegressor(),
             GradientBoostingRegressor(),
             KNeighborsRegressor(n_neighbors = 4),]

   names = ['LR','Ridge','svm','GNB','RF','GB','KNN']
  for model,name in zip(models,names):
    score = rmse_cv(model,X_train,y_train)
    print("{}    : {:.6f}, {:4f}".format(name,score.mean(),score.std()))

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    alg.fit(dtrain[predictors], dtrain['age'])
    dtrain_predictions = alg.predict(dtrain[predictors])
    if performCV:
        cv_score = -cross_val_score(alg, dtrain[predictors], dtrain['age'], cv=cv_folds, 
                                                    scoring='r2')
    print ("\nModel Report")
    print( "RMSE : %.4g" % mean_squared_error(dtrain['age'].values, dtrain_predictions))
    print( "R2 Score (Train): %f" % r2_score(dtrain['age'], dtrain_predictions)) 
    if performCV:
        print( "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score), np.min(cv_score),np.max(cv_score)))
    if printFeatureImportance:
        feat_imp = pd.Series(alg.coef_, predictors).sort_values(ascending=False)
        plt.figure(figsize=(20,4))
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

predictors = [x for x in data.columns if x not in ['age']]
lrm0 = Ridge(random_state=10)
modelfit(lrm0, data, predictors)

from sklearn.model_selection import  GridSearchCV

param  = {'alpha':[0.01, 0.1, 1,10,100],
         'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
glrm0 = GridSearchCV(estimator = Ridge(random_state=10,),
param_grid = param,scoring= 'r2' ,cv = 5,  n_jobs = -1)
glrm0.fit(X_train, y_train)
glrm0.best_params_, glrm0.best_score_

modelfit(Ridge(alpha = 0.1, solver='sag',random_state=10,), data, predictors)
