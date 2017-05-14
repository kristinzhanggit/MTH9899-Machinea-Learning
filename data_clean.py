import pickle
import numpy as np
import pandas as pd

with open("ml_finalproj_train_vF.pkl","rb") as f:
    data=pickle.load(f)

class Data_Clean(object):
    '''data: the data frame to clean, method to choose Winsor or MAD, n: clean threshold'''
    '''filter_cols: columns to filter'''
    def __init__(self, data, method, n,filter_cols):
        self.df=data
        self.n=n
        self.filter_cols=filter_cols
        self.df['to_keep']=True
        if method=="Winsor":
            self.Winsor()
        else:
            self.MAD()

    def Apply(self,row):
        if abs(row[self.col]-self.m)/self.sd>self.n:
            return False
        if row['to_keep']==False:
            return False
        return True

    '''apply Wisorization to clean data cross-sectionally'''
    def Winsor(self):
        time_stamp = self.df['timestamp'].drop_duplicates()
        result=[]
        for i in range(len(time_stamp)):
            t = time_stamp.iloc[i]
            print(t)
            filter_df = self.df.where(self.df['timestamp'].isin([t])).dropna()
            for col in self.filter_cols:
                self.m=np.mean(filter_df[col])
                self.sd=np.std(filter_df[col])
                self.col=col
                filter_df['to_keep']=filter_df.apply(self.Apply,axis=1)
            filter_df=filter_df.where(filter_df['to_keep']).dropna()
            result.append(filter_df)
        self.result_df=pd.concat(result)

    '''apply MAD to clean data cross-sectionally'''
    def MAD(self):
        time_stamp=self.df['timestamp'].drop_duplicates()
        result=[]
        for i in range(len(time_stamp)):
            t=time_stamp.iloc[i]
            filter_df=self.df.where(self.df['timestamp'].isin([t])).dropna()
            for col in self.filter_cols:
                self.m=np.median(filter_df[col])
                self.sd=np.median(abs(filter_df[col]-self.m))
                self.col=col
                filter_df['to_keep']=filter_df.apply(self.Apply,axis=1)
            filter_df=filter_df.where(filter_df['to_keep']).dropna()
            result.append(filter_df)
        self.result_df=pd.concat(result)

    '''Get cleaned data'''
    def GetData(self):
        self.result_df.drop('to_keep',axis=1,inplace=True)
        print(self.result_df.head())
        return self.result_df

index=data.columns.values
'''not_include: columns should not be filtered as they are categorical or not appropriate'''
not_include=['x29','x42','x30','x2','x6','x46','x25','x13','x28','x51','x29','timestamp','id','weight','y']
'''all cols suitable to clean as they are continuous predictors'''
continues=[col for col in index if col not in not_include]
'''construct one data clean object using mad to clean data'''
d=Data_Clean(data,"MAD",3,[continues[0]])
'''get cleaned data'''
s=d.GetData()


