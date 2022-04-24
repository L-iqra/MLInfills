from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import numpy as np
import os
import pandas as pd
import pickle
from xgboost import XGBRegressor


np.set_printoptions(precision=4)
from warnings import filterwarnings
filterwarnings("ignore")

from dash_utils import *

np.set_printoptions(precision=10)

printmd('<h2>Welcome to fundamental period prediction dashboard!</h2>')
printmd('<h3>Follow these steps to predict the fundamental period of masonary infilled RC frame.</h3>')
printmd('<b>Step 1:</b> Enter the values of all the input features.')
printmd("<b>Step 2:</b> Click to select the prediction model.")
printmd("<b>Step 3:</b> Click on Get Prediction!.")

@interact
def predict(NOSt = widgets.BoundedIntText(value = 22, min = 1, max = 22.0, step = 1, description = '<b>NOSt</b>', disabled = False),
                     OR = widgets.BoundedFloatText(value = 1., min = 0., max = 1.0, step = 0.01, description = '<b>OR</b>', disabled = False),
                     SL = widgets.BoundedFloatText(value = 7.5, min = 3, max = 10.0, step = 0.01, description = '<b>SL</b>', disabled = False),
                     K = widgets.BoundedFloatText(value = 15., min = 2.25, max = 25.0, step = 0.01, description = '<b>K</b>', disabled = False),
                     NOSp= widgets.BoundedIntText(value = 2, min = 1, max = 6.0, step = 1, description = '<b>NOSp</b>', disabled = False),
                    Model = widgets.ToggleButtons(
    options=['NN', 'XGBoost', 'Eq. (9)'],
    description = '<b>Model<b>',
    disabled = False,
    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Neural Network', 'XGBoost', 'Equation given by Charalampakis et al. [17]'],
                    style=dict(font_weight='bold'))):
    
    
    
    columns=['NOSt', 'NOSp', 'SL', 'OR', 'K']
    
    inputs = np.array([NOSt, NOSp, SL, OR, K])
    if Model == 'NN':
        scaler = pickle.load(open('./models/nn_scaler.pkl','rb'))
        inputs = (inputs - scaler.mean_[:5])/scaler.scale_[:5]
    else:pass
    pred_dict = dict()
    for i in range(len(columns)):
        pred_dict[columns[i]] = [inputs[i]]
            
         
    def on_button_clicked(b):
        with output:
            if Model == 'NN':
                mdl = pickle.load(open('./models/nn_best_model.pkl','rb'))
                tpred = mdl.predict(pd.DataFrame(pred_dict)).round(4)
                print(f'\n \n Predicted fundamental period using {Model}: ',tpred[0], 's')  
            elif Model == 'XGBoost':
        
                mdl = pickle.load(open('./models/xgb_best_model.pkl','rb'))
                tpred = mdl.predict(pd.DataFrame(pred_dict)).round(4)
                print(f'\n \n Predicted fundamental period using {Model}: ',tpred[0], 's')
            elif Model == 'Eq. (9)':
                df = pd.DataFrame(pred_dict)
                OR = df.OR
                SL = df.SL
                N  = df.NOSt
                WS = df.K
                tpred = (0.067+0.16*OR - 0.159*OR**2 + 0.002*SL**1.71)*(N**(0.95-0.004*OR+0.12*OR**2)/WS**(0.39-0.50*OR+0.110*OR**2))
                print(f'\n \n Predicted fundamental period using {Model}: ',round(tpred,4).values.item(), 's')
    button = widgets.Button(description="Get prediction",button_style = 'success',tooltip = 'Description', icon = 'compact-disc',style=dict(font_weight='bold'))
    
    output = widgets.Output()

    display(button, output)
    button.on_click(on_button_clicked)
    pass