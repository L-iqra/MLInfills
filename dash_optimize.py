from ipywidgets import interact, interactive, fixed, interact_manual, Layout
import ipywidgets as widgets
from IPython.display import Markdown, display
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from xgboost import XGBRegressor
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
np.set_printoptions(precision=4)
from warnings import filterwarnings
filterwarnings("ignore")
from dash_utils import *

                                           
printmd('<b style="font-size:35px;">The Optimization Dashboard</b>')
printmd('<b style="font-size:25px;">User Instructions:</b>')
printmd('<b><i>*Note: User only have to provide values of input features, range of OR and K and select the surrogate model for optimization.</b>')
printmd('<b>Step 1:</b> Enter the values of all the input features.')
printmd("<b>Step 2:</b> Provide the desired range of <b><i>OR</i></b> and <b><i>K</i></b>.")
printmd("<b>Step 3:</b> Provide the fundamental period to be targetted.")
printmd('<b>Step 4:</b> Choose the surrogate model.')
printmd('<b>Step 5:</b> Click on <b>Run optimization</b> to get the optimized <b><i>OR (OR_opt)</i></b>, <b><i>K (K_opt)</i></b>, and <b><i>the fundamental period after optimization (T_opt)</b>.')
# print('\n')

@interact
def optimize(NOSt = widgets.Combobox(description = '<b style="font-size:17px;">NOSt</b>', disabled = False, placeholder='Number of storeys: 1 - 22',layout=Layout(width='29%')),
             ORmin = widgets.Combobox(description = '<b style="font-size:17px;"><i>OR-min</i></b>', disabled = False, placeholder='Opening ratio \u2265 0',layout=Layout(width='29%')),
             ORmax = widgets.Combobox(description = '<b style="font-size:17px;"><i>OR-max</i></b>', disabled = False, placeholder='Opening ratio \u2264 1.0',layout=Layout(width='29%')),
             SL = widgets.Combobox(description = '<b style="font-size:17px;">SL</b>', disabled = False, placeholder='Span length (m): 3 - 7.5' ,layout=Layout(width='29%')),
             Kmin = widgets.Combobox(description = '<b style="font-size:17px;"><i>K-min</i></b>', disabled = False, placeholder='Masonry wall stiffness (x 10\u2075 kN/m) \u2265 2.25',layout=Layout(width='29%')),
             Kmax = widgets.Combobox(description = '<b style="font-size:17px;"><i>K-max</i></b>', disabled = False, placeholder='Masonry wall stiffness (x 10\u2075 kN/m) \u2264 25',layout=Layout(width='29%')),
             NOSp= widgets.Combobox(description = '<b style="font-size:17px;">NOSp</b>', disabled = False, placeholder='Number of spans: 2 - 6',layout=Layout(width='29%')),
             T = widgets.Combobox(description = '<b style="font-size:17px;"><i>T-target</i></b>', disabled = False, placeholder='Target fundamental period',layout=Layout(width='29%')),
             Model = widgets.ToggleButtons(
                                        options = ['NN', 'XGBoost', 'Eq. (9)'],
                                        description = '<b style="font-size:17px;">Model</b>',
                                        disabled = False,
                                        button_style = 'info', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltips = ['Neural Network', 'XGBoost', 'Equation given by Charalampakis et al. [17]'],
                                        style=dict(font_weight='bold'))):

    try:
        if isinstance(NOSt, str) and len(NOSt)>0:
            try:
                NOSt = int(NOSt)
                if NOSt not in range(1,23):
                    colortext('<b>NOSt</b> should be in range <b>1 - 22</b>')
            except:
                colortext('<b>NOSt</b> should be in range <b>1 - 22</b>')
        if isinstance(ORmin, str) and len(ORmin)>0:
            try:
                ORmin = float(ORmin)
                if (0 > ORmin) or (ORmin > 1.0):
                    colortext('<b>OR-min</b> should be in range <b>0 - 1</b>')
            except:
                colortext('<b>OR-min</b> should be in range <b>0 - 1</b>')
        if isinstance(ORmax, str) and len(ORmax)>0:
            try:
                ORmax = float(ORmax)
                if (0 > ORmax) or (ORmax > 1.0):
                    colortext('<b>OR-max</b> should be in range <b>0 - 1</b>')
            except:
                colortext('<b>OR-max</b> should be in range <b>0 - 1</b>')
        if isinstance(SL, str) and len(SL)>0:    
            try:
                SL = float(SL)
                if (3 > SL) or (SL > 7.5):
                    colortext('<b>SL</b> should be in range <b>3 - 7.5</b>')
            except:
                colortext('<b>SL</b> should be in range <b>3 - 7.5</b>')
        if isinstance(Kmin, str) and len(Kmin)>0:
            try:
                Kmin = float(Kmin)
                if (2.25 > Kmin) or (Kmin > 25):
                    colortext('<b>K-min</b> should be in range <b>2.25 - 25</b>')
            except:
                colortext('<b>K-min</b> should be in range <b>2.25 - 25</b>')
        if isinstance(Kmax, str) and len(Kmax)>0:
            try:
                Kmax = float(Kmax)
                if (2.25 > Kmax) or (Kmax > 25):
                    colortext('<b>K-max</b> should be in range <b>2.25 - 25</b>')
            except:
                colortext('<b>K-max</b> should be in range <b>2.25 - 25</b>')
        if isinstance(NOSp, str) and len(NOSp)>0:
            try:
                NOSp = float(NOSp)
                if (2 > NOSp) or (NOSp > 6):
                    colortext('<b>NOSp</b> should be in range <b>2 - 6</b>')
            except:
                colortext('<b>NOSp</b> should be in range <b>2 - 6</b>')
#         inputs = np.array([int(NOSt), int(NOSp), float(SL), float(OR), float(K)])
        target_var = float(T)
        inputs = [[int(NOSt), int(NOSp), int(SL), 0., 0.]]
        config = {'model':Model}
    except:
        pass
#         inputs = np.array([int(NOSt), int(NOSp), float(SL), float(OR), float(K)])
        # print('Kindly input values in specified range.')
# 
        inputs = np.array([0, 0, 0, 0, 0])
    class MyProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var = 2,
                             n_obj = 1,
                             n_constr = 0,
                             # lOWER AND UPPER BOUNDS OF mu AND R
                             xl = np.array([ORmin, Kmin]), #lower limits of op and K
                             xu = np.array([ORmax, Kmax])) #upper limits of op and K
        def _evaluate(self, x, out, *args, **kwargs):
           # NOSt = x[0]
            #SL = x[1]>
            OR = x[0]
            K = x[1]
            f1 = fitness_func_model(OR,K,inputs) #change fitness function according to model
            out["F"] = [f1]
    termination = SingleObjectiveDefaultTermination(
                x_tol=1e-7,
                cv_tol=1e-7,
                f_tol=1e-7,
                nth_gen=50)
    try:
        target_var = float(T)
        inputs = [[int(NOSt), int(NOSp), float(SL), 0., 0.]]
        config = {'model':Model}
    except:
        # print('Kindly enter the values in the specified range.')
        target_var = T
        inputs = [[NOSt, NOSp, SL, 0., 0.]]
        config = {'model':Model}
    def fitness_func_model(OR,K,inputs):
        NOSt = inputs[0][0]
        NOSp = inputs[0][1]
        SL = inputs[0][2]
        OR = OR
        K = K
        inp = np.array([NOSt,NOSp,SL,OR,K]).reshape(1,-1)
        pred_var =  predictor(inp, config) #mdl.predict((inp-scaler.mean_[:-1])/scaler.scale_[:-1])[0]            
        fitness = 100*np.abs(target_var-pred_var)/target_var
        return fitness
    def on_button_clicked(b):
        with output:
            colortext(f'<b>Running optimization using : {Model}</b>','orange')
            # def fitness_func_model(OR,K,inputs):
            #     NOSt = inputs[0][0]
            #     NOSp = inputs[0][1]
            #     SL = inputs[0][2]
            #     OR = OR
            #     K = K
            #     inp = np.array([NOSt,NOSp,SL,OR,K]).reshape(1,-1)
            #     pred_var =  predictor(inp, config) #mdl.predict((inp-scaler.mean_[:-1])/scaler.scale_[:-1])[0]            
            #     fitness = 100*np.abs(target_var-pred_var)/target_var
            #     return fitness
            problem = MyProblem()
            algorithm = GA(
                pop_size=50,
                eliminate_duplicates=True)
            res = minimize(problem,
                           algorithm,termination,
                           seed=11351,
                           verbose=False,save_history=True)
            # print(res)
            inputs[0][3] = res.X[0]
            inputs[0][4] = res.X[1]
            op = predictor(inputs, config)
            colortext('<b>Optimization completed!</b>', 'green')
            inputs[0].extend([target_var, op])  #inserting target time period and optimization results 
            colnames = ['NOSt','NOSp','SL', 'OR_opt', 'K_opt', 'T_tgt', 'T_opt']
            df = pd.DataFrame(inputs, columns=colnames,index=['Output']).round(3)
            display(df)
            begin = False
    button = widgets.Button(description="Run optimization", tooltip = 'Description', icon = 'compact-disc',style=dict(font_weight='bold', button_color='lightpink'),
                           layout=Layout(width='18%', height='40px'))
    button.add_class('myclass')
                                           
                                           
                                           
    output = widgets.Output()
    display(button, output)
    button.on_click(on_button_clicked)
    pass