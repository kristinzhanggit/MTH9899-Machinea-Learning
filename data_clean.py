import numpy as np
import pandas as pd


class Data_Clean(object):
    def __init__(self, data, method, n,filter_cols,not_include):
        self.df=data
        print("original data:",len(data))
        self.n=n
        self.filter_cols=filter_cols
        self.df['to_keep']=0
        self.not_include=not_include
        if method=="Winsor":
            self.Winsor()
        else:
            self.MAD()

    def Winsor(self):
        for col in self.filter_cols:
            if col in self.not_include: continue
            m=np.mean(self.df[col])
            sd=np.std(self.df[col])
            self.df['to_keep']=np.maximum(self.df['to_keep'].values.T,np.absolute((self.df[col]-m)/sd).values.T)
        self.df=self.df.where(self.df['to_keep']<self.n).dropna()

    def MAD(self):
        for col in self.filter_cols:
            if col in self.not_include: continue
            m=np.median(self.df[col])
            sd=np.median(abs(self.df[col]-m))
            self.df['to_keep']=np.maximum(self.df['to_keep'].values.T,np.absolute((self.df[col]-m)/sd).values.T)
        self.df=self.df.where(self.df['to_keep']<self.n).dropna()

    def GetData(self):
        self.df.drop('to_keep',axis=1,inplace=True)
        print("data cleaned: ",len(self.df))
        return self.df