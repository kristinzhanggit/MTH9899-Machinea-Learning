import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt

with open("ml_finalproj_train_vF.pkl","rb") as f:
    data=pickle.load(f)

not_include1=['x29','x42','x30','x2','x6','x46','x25','x13','x28','x51','timestamp','id']
data=data.drop('id',axis=1)
y=data['y'].copy()
y_fixed_std=np.std(y)
#y_fixed_std=1
y=y/y_fixed_std
X=data.drop(['weight','y','timestamp'],axis=1)
weight=data['weight']

def Normalize(data,not_include=not_include1):
    for col in data.columns.values:
        if col in not_include: continue
        data[col]=(data[col]-np.mean(data[col]))/np.std(data[col])
    return data

def LassoSelect(X,y,cv_fold):
    estimator=LassoCV(cv=cv_fold).fit(X,y)
    selector=SelectFromModel(estimator,prefit=True)
    return X.columns.values[selector.get_support()]

X=Normalize(X)
cols_use=LassoSelect(X,y,5)
total_data=pd.concat([X,y,weight],axis=1)