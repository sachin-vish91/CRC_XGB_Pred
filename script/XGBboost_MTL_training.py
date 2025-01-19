### run XGB MTL model

import numpy as np
import pandas as pd
import  scipy as scp
import random, sys
import xgboost as xgb
import sys, time
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import distance
import pickle

# Custom function for RMSE and R2 calculations
def rmse(a,b):
    return np.sqrt(np.sum((a-b)**2)/len(a))

def r2(a,b):
    return 1-np.sum((a-b)**2)/np.sum((a-np.mean(a))**2)

cell_data_names = {'BR:MCF7':'MCF7', 'BR:MDA-MB-231':'MDA-MB-231_ATCC', 'BR:HS 578T':'HS_578T', 
                   'BR:BT-549':'BT-549', 'BR:T-47D':'T-47D', 'CNS:SF-268':'SF-268',
                   'CNS:SF-295':'SF-295', 'CNS:SF-539':'SF-539', 'CNS:SNB-19':'SNB-19', 
                   'CNS:SNB-75':'SNB-75', 'CNS:U251':'U251', 'CO:COLO 205':'COLO_205', 
                   'CO:HCC-2998':'HCC-2998', 'CO:HCT-116':'HCT-116', 'CO:HCT-15':'HCT-15',
                   'CO:HT29':'HT29', 'CO:KM12':'KM12', 'CO:SW-620':'SW-620', 
                   'LE:CCRF-CEM':'CCRF-CEM', 'LE:HL-60(TB)':'HL-60(TB)', 'LE:K-562':'K-562', 
                   'LE:MOLT-4':'MOLT-4', 'LE:RPMI-8226':'RPMI-8226', 'LE:SR':'SR', 
                   'ME:LOX IMVI':'LOX_IMVI','ME:MALME-3M':'MALME-3M', 'ME:M14':'M14', 
                   'ME:SK-MEL-2':'SK-MEL-2', 'ME:SK-MEL-28':'SK-MEL-28', 'ME:SK-MEL-5':'SK-MEL-5',
                   'ME:UACC-257':'UACC-257', 'ME:UACC-62':'UACC-62', 'ME:MDA-MB-435':'MDA-MB-435', 
                   'ME:MDA-N':'', 'LC:A549/ATCC':'A549_ATCC', 'LC:EKVX':'EKVX', 
                   'LC:HOP-62':'HOP-62', 'LC:HOP-92':'HOP-92', 'LC:NCI-H226':'NCI-H226',
                   'LC:NCI-H23':'NCI-H23', 'LC:NCI-H322M':'NCI-H322M', 'LC:NCI-H460':'NCI-H460', 
                   'LC:NCI-H522':'NCI-H522', 'OV:IGROV1':'IGROV1', 'OV:OVCAR-3':'OVCAR-3', 
                   'OV:OVCAR-4':'OVCAR-4', 'OV:OVCAR-5':'OVCAR-5', 'OV:OVCAR-8':'OVCAR-8', 
                   'OV:SK-OV-3':'SK-OV-3', 'OV:NCI/ADR-RES':'NCI_ADR-RES', 'PR:PC-3':'PC-3', 
                   'PR:DU-145':'DU-145', 'RE:786-0':'786-0', 'RE:A498':'A498',
                   'RE:ACHN':'ACHN', 'RE:CAKI-1':'CAKI-1', 'RE:RXF 393':'RXF_393', 
                   'RE:SN12C':'SN12C', 'RE:TK-10':'TK-10', 'RE:UO-31':'UO-31'}

a = pd.read_excel('../profile_data/RNA__Affy_HG_U133_Plus_2.0_RMA.xls', skiprows=10, header=0, na_values='-')
gexp = np.array(a.groupby("Gene name d", as_index=False).mean().loc[:,'BR:MCF7':].T, dtype=np.float32)
cells_in_gexp = [cell_data_names[i] for i in a.columns[9:]]

# only in case of GeneExp profile because it does not have data for ME:MDA-N cell line
gexp = np.delete(gexp,33,0)
cells_in_gexp.remove('')

len(cells_in_gexp)

spearman_kernel = distance.pdist(gexp, metric=lambda x,y:spearmanr(x,y)[0])
spearman_kernel = distance.squareform(spearman_kernel)
spearman_kernel += np.eye(spearman_kernel.shape[0])

best_indices = list(range(len(gexp[0])))

start = time.time()

traindata = pd.read_csv('../cell_line_smile/'+cells_in_gexp[0]+'.train.csv')
y_train = traindata.NLOGGI50_N.values
traindata.drop(columns= ['PANEL','CELL','STD_SMILE','NLOGGI50_N','NSC','SMILE'], inplace=True)
x_train = traindata.values
x_train = np.hstack([x_train, np.tile(spearman_kernel[0], (len(x_train),1))])

for c in cells_in_gexp[1:]:
    traindata = pd.read_csv('../cell_line_smile/'+c+'.train.csv')
    y_train1 = traindata.NLOGGI50_N.values
    traindata.drop(columns= ['PANEL','CELL','STD_SMILE','NLOGGI50_N','NSC','SMILE'], inplace=True)
    t = traindata.values
    t = np.hstack([t, np.tile(spearman_kernel[cells_in_gexp.index(c)], (len(t),1))])
    x_train = np.vstack([x_train, t])
    y_train = np.concatenate([y_train, y_train1])

print(x_train.shape)

XGB_model = xgb.XGBRegressor(max_depth=10, learning_rate=0.05, n_estimators=1000, colsample_bytree=0.9, n_jobs=35)

XGB_model.fit(x_train, y_train, eval_metric='rmse')
print('Model building took', int(time.time() - start)//60, 'min.' )

# Save trained XGBoost model
model_path = '../MTL_XGB.dat'

# Save the model using pickle
with open(model_path, 'wb') as file:
    pickle.dump(XGB_model, file)
