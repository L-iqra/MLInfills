import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd
import pickle
from xgboost import XGBRegressor
from warnings import filterwarnings
filterwarnings("ignore")

np.set_printoptions(precision=4)

np.set_printoptions(precision=10)

def load_model(config):
    
    if config['model'] == 'XGBoost':
        mdl = pickle.load(open('./models/xgb_best_model.pkl','rb'))
        return mdl, None
    
    
    elif config['model'] == 'NN':
        mdl = pickle.load(open('./models/nn_best_model.pkl','rb'))
        scaler = pickle.load(open('./models/nn_scaler.pkl','rb'))
        return mdl, scaler
        
        
    elif config['model']=='equation':
        def use_eqn(df=None):
            """
            Function for making predictions only for single structure
            """
            OR = df.OR
            SL = df.SL
            N = df.NOSt
            WS = df.K
            # print(OR,SL,N,WS)
#             print('Loading completed')
            T = (0.067+0.16*OR - 0.159*OR**2 + 0.002*SL**1.71)*(N**(0.95-0.004*OR+0.12*OR**2)/WS**(0.39-0.50*OR+0.110*OR**2))
            return T.values[0]
        def use_eqn_full(df=None):
            """
            Function for making predictions on more than one structure
            """
            OR = df.OR
            SL = df.SL
            N = df.NOSt
            WS = df.K
            T = (0.067+0.16*OR - 0.159*OR**2 + 0.002*SL**1.71)*(N**(0.95-0.004*OR+0.12*OR**2)/WS**(0.39-0.50*OR+0.110*OR**2))
#             print('Loading completed')
            return T
        return use_eqn, use_eqn_full
      
      
config = {'model':'Eq. (9)'}
def predictor(input_info, config = config):
    ''' 
    Predicts the outpus for a given datapoint
    Takes input of list and model name
    predictor(input_info, config = config)
    '''
    model_name = config['model']
    
    if model_name == 'XGBoost':
        mdl,_=load_model(config)
    
        columns=['NOSt', 'NOSp', 'SL', 'OR', 'K']
        input_info = np.array(input_info)
        pred_dict = dict()
        for i in range(len(columns)):
            pred_dict[columns[i]] = [input_info[0][i]]
        tp = mdl.predict(pd.DataFrame(pred_dict)).round(4).item()
        
    elif model_name == 'NN':
        mdl, scaler = load_model(config)
        tp = mdl.predict((input_info-scaler.mean_[:-1])/scaler.scale_[:-1])[0]
        
    elif model_name=='Eq. (9)':
        def use_eqn(df=None):
            """
            Function for making predictions only for single structure
            """
            OR = df.OR
            SL = df.SL
            N = df.NOSt
            WS = df.K
            # print(OR,SL,N,WS)
            T = (0.067+0.16*OR - 0.159*OR**2 + 0.002*SL**1.71)*(N**(0.95-0.004*OR+0.12*OR**2)/WS**(0.39-0.50*OR+0.110*OR**2))
#             print('Loading completed')
            return T.values[0]
        dfinp=pd.DataFrame(data=input_info,columns=['NOSt', 'NOSp', 'SL', 'OR', 'K'])
        tp = use_eqn(dfinp)
    return tp