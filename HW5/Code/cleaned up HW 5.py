#cleaned up HW 5
import os 
import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganGenerator
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#change the path as needed
datapath = "D:/MyPythoncode/HW5/Data/lipophilicity.csv"
mfp_gen = GetMorganGenerator(radius=2, fpSize=2048)
#fingerprint generator
def gen_mfp_def(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(mfp_gen.GetFingerprint(mol))
#Main
rawdata = pd.read_csv(datapath)
rawdata['mfp'] = rawdata['smiles'].apply (gen_mfp_def)
#I understand that this is a conversion to a 1d array, but the method feels weird, provided by stack overflow
m_fp = np.array(list(rawdata['mfp'].values))
x_train, x_test, y_train, y_test = train_test_split(m_fp,rawdata['exp'],test_size=0.3)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
#parse (a british man on youtube showed me this) 
parser = argparse.ArgumentParser (description="Lipophilicity morgan fingerprint.")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=50)
parser.add_argument("--max_iter", type=int, default=1000)
args = parser.parse_args()
#standard scaler
scaler = StandardScaler()
#I can't believe I actually got .ravel to work, I'm celebrating
y_train_scale = scaler.fit_transform(y_train.reshape(-1,1)).ravel()
mlp = MLPRegressor(hidden_layer_sizes=(args.n_estimators,args.max_depth), max_iter=args.max_iter)
mlp.fit(x_train,y_train_scale)

y_predict_scaled = mlp.predict(x_test)
y_predict = scaler.inverse_transform(y_predict_scaled.reshape(1,-1)).ravel()
RMSE = np.sqrt(mean_squared_error(y_test,y_predict))
Conda_Enviornment = os.getenv("CONDA_DEFAULT_ENV")

with open('results.txt', "w") as r:
    r.write(f"RMSE: {RMSE}")
    r.write(f"    Conda Enviornment: {Conda_Enviornment}")
    r.write(f"\n Hyperperameters:\n N Estimator: {args.n_estimators}")
    r.write(f"    Max Depth: {args.max_depth}")
    r.write(f"    Max Iter: {args.max_iter}")