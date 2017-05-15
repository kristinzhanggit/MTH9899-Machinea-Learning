import pickle
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV


with open("ml_finalproj_train_vF.pkl","rb") as f:
    data=pickle.load(f)

not_include1=['x29','x42','x30','x2','x6','x46','x25','x13','x28','x51','timestamp','id']
data=data.drop('id',axis=1)

'''normalize non-categorical data'''
def Normalize(data,not_include=not_include1):
    time_stamp = data['timestamp'].drop_duplicates()
    print(len(time_stamp))
    for i in range(len(time_stamp)):
        t = time_stamp.iloc[i]
        for col in data.columns.values:
            if col in not_include: continue
            data.loc[data['timestamp'].isin([t]),col]=(data.loc[data['timestamp'].isin([t]),col]-np.mean(data.loc[data['timestamp'].isin([t]),col]))/np.std(data.loc[data['timestamp'].isin([t]),col])
    return data

'''use lasso to select features'''
def LassoSelect(data, cv_fold):
    X=data.drop('y',axis=1)
    y=data['y']
    estimator=LassoCV(cv=cv_fold).fit(X,y)
    selector=SelectFromModel(estimator,prefit=True)
    print(X.columns.values[selector.get_support()])
    return selector

data=Normalize(data)
LassoSelect(data,5)