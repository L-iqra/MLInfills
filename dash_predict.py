from ipywidgets import interact, interactive, fixed, interact_manual, Layout
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

printmd('<b style="font-size:35px;">The Fundamental Period Prediction Dashboard</b>')
printmd('<b style="font-size:25px;">User Instructions:</b>')
printmd('<b><i>*Note: User only have to provide values of input features and select the prediction model for output.</b>')
printmd('<b>Step 1:</b> Enter the values of all the input features.')
printmd("<b>Step 2:</b> Click to select the prediction model.")
printmd("<b>Step 3:</b> Click on <b>Get Prediction</b> to obtain <b>predicted fundamental period (<i>T<sub>predicted</sub></i>).")


def colortext(text='text', color='red'):
    printmd(f"<font color='{color}'>{text}</font>")
@interact
def predict(NOSt = widgets.Combobox(description = '<b style="font-size:17px;">NOSt</b>', disabled = False,placeholder='Number of storeys: 1 - 22',layout=Layout(width='29%')),
                     OR = widgets.Combobox(description = '<b style="font-size:17px;">OR</b>', disabled = False, placeholder='Opening ratio: 0 - 1',layout=Layout(width='29%')),
                     SL = widgets.Combobox(description = '<b style="font-size:17px;">SL</b>', disabled = False, placeholder='Span length (m): 3 - 7.5 ',layout=Layout(width='29%')),
                     K = widgets.Combobox(description = '<b style="font-size:17px;">K</b>', disabled = False, placeholder='Masonry wall stiffness (x 10\u2075 kN/m): 2.25 - 25',layout=Layout(width='29%')),
                     NOSp= widgets.Combobox(description = '<b style="font-size:17px;">NOSp</b>', disabled = False, placeholder='Number of spans: 2 - 6',layout=Layout(width='29%')),
                    Model = widgets.ToggleButtons(
    options=['NN', 'XGBoost', 'Eq. (9)'],
    description = '<b style="font-size:17px;">Model</b>',
    disabled = False,
    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Neural Network', 'XGBoost', 'Equation given by Charalampakis et al. [17]'],
                    style=dict(font_weight='bold'))):
    
    
    
    columns=['NOSt', 'NOSp', 'SL', 'OR', 'K']
#     try:
#         inputs = np.array([int(NOSt), int(NOSp), float(SL), float(OR), float(K)])
    try:
        if isinstance(NOSt, str) and len(NOSt)>0:
            try:
                NOSt = int(NOSt)
                if NOSt not in range(1,23):
                    colortext('<b>NOSt</b> should be in range <b>1 - 22</b>')
            except:
                colortext('<b>NOSt</b> should be in range <b>1 - 22</b>')
                
        if isinstance(OR, str) and len(OR)>0:
            try:
                OR = float(OR)
                if (0 > OR) or (OR > 1.0):
                    colortext('<b>OR</b> should be in range <b>0 - 1</b>')

            except:
                colortext('<b>OR</b> should be in range <b>0 - 1</b>')
                
        if isinstance(SL, str) and len(SL)>0:    
            try:
                SL = float(SL)
                if (2 > SL) or (SL > 7.5):
                    colortext('<b>SL</b> should be in range <b>3 - 7.5</b>')

            except:
                colortext('<b>SL</b> should be in range <b>3 - 7.5</b>')
                
        if isinstance(K, str) and len(K)>0:
            try:
                K = float(K)
                if (2.25 > K) or (K > 25):
                    colortext('<b>K</b> should be in range <b>2.25 - 25</b>')

            except:
                colortext('<b>K</b> should be in range <b>2.25 - 25</b>')
                
        if isinstance(NOSp, str) and len(NOSp)>0:
            try:
                NOSp = float(NOSp)
                if (2 > NOSp) or (NOSp > 6):
                    colortext('<b>NOSp</b> should be in range <b>2 - 6</b>')
            except:
                colortext('<b>NOSp</b> should be in range <b>2 - 6</b>')
        
        inputs = np.array([int(NOSt), int(NOSp), float(SL), float(OR), float(K)])
    except:
        
#         pass
#         inputs = np.array([int(NOSt), int(NOSp), float(SL), float(OR), float(K)])
#         print('Kindly input values in specified range.')
# 
        inputs = np.array([0, 0, 0, 0, 0])
    try:
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
                    opmessage = f'<b><i>T<sub>predicted</sub></i> using <i>{Model}: {tpred[0]} s</i></b>'
                    printmd(opmessage) 
                elif Model == 'XGBoost':

                    mdl = pickle.load(open('./models/xgb_best_model.pkl','rb'))
                    tpred = np.array(mdl.predict(pd.DataFrame(pred_dict)).round(4)).round(4).item()
                    tpred = round(tpred, 4)
                    opmessage = f'<b><br><i>T<sub>predicted</sub></i> using <i>{Model}: {tpred} s</i></b>'
                    printmd(opmessage)
                elif Model == 'Eq. (9)':
                    df = pd.DataFrame(pred_dict)
                    OR = df.OR
                    SL = df.SL
                    N  = df.NOSt
                    WS = df.K
                    tpred = (0.067+0.16*OR - 0.159*OR**2 + 0.002*SL**1.71)*(N**(0.95-0.004*OR+0.12*OR**2)/WS**(0.39-0.50*OR+0.110*OR**2))
                    opmessage = f'<b><br><i>T<sub>predicted</sub></i> using <i>{Model}: {round(tpred,4).values.item()} s</i></b>'
                    printmd(opmessage)

        button = widgets.Button(description='Get prediction',tooltip = 'Description', icon = 'compact-disc',style=dict(font_weight='bold', button_color='lightpink'),
                               layout=Layout(width='15%', height='40px'))
        button.add_class('myclass')
        output = widgets.Output()
#         printmd('<br>')
        display(button, output)
        button.on_click(on_button_clicked)
    except ValueError:
        pass