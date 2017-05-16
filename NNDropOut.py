from data_clean import *
from Feature_selection import *
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from scipy.stats import binom
import warnings

warnings.filterwarnings('ignore')
n=len(total_data)

class RandomDropout(object):
    def __init__(self, total_data, cols_use, first_num, second_num, not_include=not_include1, cv_folds=5):
        self.total_data=total_data
        self.y_pred=np.zeros(len(self.total_data))
        self.out_size=int(len(self.total_data)/cv_folds)
        self.cv_folds=cv_folds
        self.first_min, self.first_max=first_num
        self.second_min, self.second_max=second_num
        self.to_remove=[col for col in self.total_data.columns.values if col not in cols_use]
        self.cols_use=cols_use
        self.not_include=not_include

    def Dropout_set(self):
        self.layer_map={}
        first_dff=self.first_max-self.first_min
        second_dff=self.second_max-self.second_min
        for i in range(first_dff+1):
            for j in range(second_dff+1):
                key_val=(self.first_min+i,self.second_min+j)
                self.layer_map[key_val]=binom.pmf(i,first_dff,0.5)*binom.pmf(j,second_dff,0.5)

    def Regress(self,key_val):
        for i in range(self.cv_folds):
            start=i*self.out_size
            end=start+self.out_size
            out_sample=self.total_data.iloc[start:end,:]
            in_sample=pd.concat([self.total_data.iloc[:start,:],self.total_data.iloc[end:,:]])
            in_clean=Data_Clean(in_sample,"Winsor",4,self.cols_use,self.not_include).GetData()
            y_clean=in_clean['y']
            x_clean=in_clean.drop(self.to_remove,axis=1)
            x_out=out_sample.drop(self.to_remove,axis=1)
            y_out=out_sample['y']
            regressor=MLPRegressor(hidden_layer_sizes=key_val,activation='logistic',max_iter=80000)
            regressor=regressor.fit(x_clean,y_clean)
            y_pred=regressor.predict(x_out)
            for j in range(len(y_out)):
                val_add=y_pred[j]*self.layer_map[key_val]
                self.y_pred[start+j]+=val_add

    def FitModel(self):
        for key_val in self.layer_map.keys():
            self.Regress(key_val)
        r2=r2_score(self.total_data['y'],self.y_pred,self.total_data['weight'])
        print(r2)

    def Predict(self,sample_data):
        predict_y=np.zeros(len(sample_data))
        in_clean = Data_Clean(self.total_data, "Winsor", 4, self.cols_use, self.not_include).GetData()
        y_clean = in_clean['y']
        x_clean = in_clean.drop(self.to_remove, axis=1)
        x_out = sample_data.drop(self.to_remove, axis=1)
        for key_val in self.layer_map.keys():
            regressor = MLPRegressor(hidden_layer_sizes=key_val, activation='logistic', max_iter=80000)
            regressor = regressor.fit(x_clean, y_clean)
            y_pred = regressor.predict(x_out)
            predict_y+=y_pred*self.layer_map[key_val]
        return predict_y

def run_model(your_model, data):
    return your_model.Predict(data)