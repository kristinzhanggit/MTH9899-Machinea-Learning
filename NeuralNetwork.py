from data_clean import *
from Feature_selection import *
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from scipy.stats import binom
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings('ignore')
n=len(total_data)
out_sample=int(0.2*n)

score_positive={}
for i in range(2,85):
    score_positive[i]=[]

for j in range(10):
  print("round:"+str(j))
  total_data=shuffle(total_data)
  out_s = total_data.iloc[:out_sample, :]
  in_s = total_data.iloc[out_sample:, :]
  data_clean = Data_Clean(in_s, "Winsor", 4, cols_use, not_include1).GetData()
  y_clean = data_clean['y']
  to_remove = [col for col in data_clean.columns.values if col not in cols_use]
  X_clean = data_clean.drop(to_remove, axis=1)
  X_out = out_s.drop(to_remove, axis=1)
  y_out = out_s['y']
  for i in range(2,85):
    regressor=MLPRegressor(hidden_layer_sizes=(i,),activation="logistic",max_iter=100000)
    regressor=regressor.fit(X_clean,y_clean)
    y_pred=regressor.predict(X_out)
    score2=r2_score(y_out,y_pred,out_s['weight'])
    if score2>0:
        score_positive[i].append(score2)
        print(i,score2)

print(score_positive)


'''class RandomDropout(object):
    def __init__(self, total_data, method, n, cols_use, first_num, second_num, third_num, cv_folds=5):
        self.total_data=total_data
        self.cv_folds=cv_folds
        self.data_clean=Data_Clean(total_data,method,n,cols_use)
        self.first=first_num
        self.second=second_num
        self.third=third_num
        self.to_remove=[col for col in self.data_clean.columns.values if col not in cols_use]

    def Dropout_set(self):
        self.layer_map={}
        for f in range(self.first+1):
            for s in range(self.second+1):
                for t in range(self.third+1):
                    p=binom.pmf(f,self.first,0.5)*binom.pmf(s,self.second,0.5)*binom.pmf(t,self.third,0.5)
                    key_arr=[]
                    key_val=None
                    if f>0: key_arr.append(f)
                    if s>0: key_arr.append(s)
                    if t>0: key_arr.append(t)
                    if len(key_arr)==0: key_val=(1)
                    elif len(key_arr)==1: key_val=(key_arr[0])
                    elif len(key_arr)==2: key_val=(key_arr[0],key_arr[1])
                    else: key_val=(key_arr[0],key_arr[1],key_arr[2])
                    if key_val in self.layer_map:
                        self.layer_map[key_val]+=p
                    else:
                        self.layer_map[key_val]=p'''
