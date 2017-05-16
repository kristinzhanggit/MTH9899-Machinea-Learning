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
'''tune 1-layer NN'''
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
    if score2>0.0005:
        score_positive[i].append(score2)
        print(i,score2)

print(score_positive)

score_positive2={}
for i in range(2,36):
  for j in range(1,23):
    key_val=(i,j)
    score_positive2[key_val]=[]
'''tune 2-layer NN'''
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
  for key_val in score_positive2.keys():
    regressor=MLPRegressor(hidden_layer_sizes=key_val,activation="logistic",max_iter=80000)
    regressor=regressor.fit(X_clean,y_clean)
    y_pred=regressor.predict(X_out)
    score2=r2_score(y_out,y_pred,out_s['weight'])
    if score2>0.0001:
        score_positive2[key_val].append(score2)
        print(key_val,score2)

print(score_positive2)
