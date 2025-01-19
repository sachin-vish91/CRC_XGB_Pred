# Prediciton on XGBboost MTL trained model.
### Imports
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import distance
import glob, argparse
import sys, time, os, random
import  scipy as scp
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import tarfile
import pickle
import joblib
from itertools import chain
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdmolops
from molvs import Standardizer
from rdkit.Chem import SaltRemover
import concurrent.futures

### Load input file for prediction file contails ID, smiles
df = pd.read_csv("../input_smiles.csv")
mol_list = df.iloc[:,1]
smiles = mol_list

### Generate feature for the input file
def Morgan_fingerprints(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol)
        s = Standardizer()
        mol = s.standardize(mol)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256) #Morgan fingerprints
        fingerprint_desc = list(fp1)
        fingerprint_desc.append(rdMolDescriptors.CalcTPSA(mol)) #total polar surface area
        fingerprint_desc.append(rdMolDescriptors.CalcExactMolWt(mol)) #molecular weight 
        fingerprint_desc.append(rdMolDescriptors.CalcCrippenDescriptors(mol)[0]) #logP
        fingerprint_desc.append(rdMolDescriptors.CalcNumAliphaticRings(mol)) #number of aliphatic ring
        fingerprint_desc.append(rdMolDescriptors.CalcNumAromaticRings(mol)) #number of aromatic ring
        fingerprint_desc.append(rdMolDescriptors.CalcNumHBA(mol)) #Number of hydrongen bond acceptor
        fingerprint_desc.append(rdMolDescriptors.CalcNumHBD(mol)) #Number of hydrongen bond doner
        fingerprint_desc.append(Chem.MolToSmiles(mol))
    except:
        fingerprint_desc = ''
    return fingerprint_desc

### code for the parallel computing multiple molecules at a time
with concurrent.futures.ProcessPoolExecutor() as executor:
    future = executor.map(Morgan_fingerprints, smiles)
    for result in future:
        file_object = open('Fingerprints.txt', 'a')
        result = str(result).replace('[', '')
        result = str(result).replace(']', '')
        result = str(result).replace("'", '')
        file_object.write(result+'\n')
        file_object.close()

df = pd.read_csv('Fingerprints.txt', header=None)
Fig = list("F_{0}".format(i) for i in range(1,257))
Pro = list("P_{0}".format(i) for i in range(1,61))
PH = ["MW","TPSA","LOGP","NAR","NARR","HBA","BHD"]
smile = ['smiles']
A = [Fig,PH,smile]

col_names = list(chain(*A))

### Normalize the data this will return the value between 0 and 1

def Normalization(x,overall_min,overall_max):
    #overall_min = 45 # specify the global minimun from all the datasets
    #overall_max = 800 # specify the global maximum from all the datasets
    x = (x - overall_min)/(overall_max - overall_min)
    
    return x

df.iloc[:,256] =  Normalization(x=df.iloc[:,256], overall_min=0, overall_max=1288.39)
df.iloc[:,257] =  Normalization(x=df.iloc[:,257], overall_min=32.03, overall_max=3351.54)
df.iloc[:,258] =  Normalization(x=df.iloc[:,258], overall_min=-18.69, overall_max=41.84)
df.iloc[:,259] =  Normalization(x=df.iloc[:,259], overall_min=0, overall_max=22)
df.iloc[:,260] =  Normalization(x=df.iloc[:,260], overall_min=0, overall_max=19)
df.iloc[:,261] =  Normalization(x=df.iloc[:,261], overall_min=0, overall_max=79)
df.iloc[:,262] =  Normalization(x=df.iloc[:,262], overall_min=0, overall_max=47)

col_names.remove('smiles')
test2 = df.loc[:,col_names]

### Load profile data
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
                   'ME:MDA-N':'MDA-N', 'LC:A549/ATCC':'A549_ATCC', 'LC:EKVX':'EKVX', 
                   'LC:HOP-62':'HOP-62', 'LC:HOP-92':'HOP-92', 'LC:NCI-H226':'NCI-H226',
                   'LC:NCI-H23':'NCI-H23', 'LC:NCI-H322M':'NCI-H322M', 'LC:NCI-H460':'NCI-H460', 
                   'LC:NCI-H522':'NCI-H522', 'OV:IGROV1':'IGROV1', 'OV:OVCAR-3':'OVCAR-3', 
                   'OV:OVCAR-4':'OVCAR-4', 'OV:OVCAR-5':'OVCAR-5', 'OV:OVCAR-8':'OVCAR-8', 
                   'OV:SK-OV-3':'SK-OV-3', 'OV:NCI/ADR-RES':'NCI_ADR-RES', 'PR:PC-3':'PC-3', 
                   'PR:DU-145':'DU-145', 'RE:786-0':'786-0', 'RE:A498':'A498',
                   'RE:ACHN':'ACHN', 'RE:CAKI-1':'CAKI-1', 'RE:RXF 393':'RXF_393', 
                   'RE:SN12C':'SN12C', 'RE:TK-10':'TK-10', 'RE:UO-31':'UO-31'}

args_profile_folder = '../profile_data'
args_p = 'GeneExp'
f = glob.glob(args_profile_folder+'/'+args_p+'/*.xls')
a = pd.read_excel(f[0], skiprows=10, header=0, na_values='-')

a.dropna(subset=a.columns[9:], axis=0, inplace=True)

### Transforming the data using groupby and calculating mean of duplicate entry
tmp = np.array(a.groupby('Gene name d', as_index=False).mean().loc[:,'BR:MCF7':].T, dtype=np.float32)
cells_from_profile = [cell_data_names[i] for i in a.columns[9:]]

### Important pre-processing of GeneExp Profile because in this profile the data is not available for MDA-N cell line
if args_p == 'GeneExp':
    tmp = np.delete(tmp,33,0)
    cells_from_profile.remove('MDA-N')

profile = distance.pdist(tmp, metric=lambda x,y:spearmanr(x,y)[0])
profile = distance.squareform(profile)
profile += np.eye(profile.shape[0])

prof = profile
prof = pd.DataFrame(prof)

cells_from_profile.index('KM12')
pp = (prof).to_numpy()
test = np.hstack([test2, np.tile(pp[cells_from_profile.index('KM12')], (len(test2),1))])

### Load trained model

XGBboost = pickle.load(open('../MTL_XGB.dat', "rb"))
preds = XGBboost.predict(test)

### Save predictions to a CSV file
output = pd.DataFrame({'Prediction': preds})
output.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'.")
