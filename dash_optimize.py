from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt

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




@interact
def optimize(NOSt = widgets.BoundedIntText(value = 22, min = 1, max = 22.0, step = 1, description = 'NOSt', disabled = False),
             ORmin = widgets.BoundedFloatText(value = 0.25, min = 0., max = 1.0, step = 0.001, description = 'OR-min', disabled = False),
             ORmax = widgets.BoundedFloatText(value = 0.75, min = 0., max = 1.0, step = 0.001, description = 'OR-max', disabled = False),
             SL = widgets.BoundedFloatText(value = 7.5, min = 3, max = 10.0, step = 0.01, description = 'SL', disabled = False),
             Kmin = widgets.BoundedFloatText(value = 2.25, min = 2.25, max = 25.0, step = 0.01, description = 'K-min', disabled = False),
             Kmax = widgets.BoundedFloatText(value = 25., min = 2.25, max = 25.0, step = 0.01, description = 'K-max', disabled = False),
             NOSp= widgets.BoundedIntText(value = 2, min = 1, max = 6.0, step = 1, description = 'NOSp', disabled = False),
             T = widgets.BoundedFloatText(value = 1.5, min = 0.0, max = 10.0, step = 0.0001, description = 'T-target', disabled = False),
             Model = widgets.ToggleButtons(
                                        options = ['NN', 'XGBoost', 'Eq. (9)'],
                                        description = 'Model',
                                        disabled = False,
                                        button_style = 'info', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltips = ['Neural Network', 'XGBoost', 'Equation given by Charalampakis et al. [17]'])):
            # optimizeme = widgets.ToggleButton(
            #                             value = False,
            #                             description = 'Run optimization',
            #                             disabled = False,
            #                             button_style = 'success',
            #                             tooltip = 'Description',
            #                             icon = 'compact-disc')):
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
            #SL = x[1]
            OR = x[0]
            K = x[1]
            f1 = fitness_func_model(OR,K,inputs) #change fitness function according to model
            out["F"] = [f1]
            
    termination = SingleObjectiveDefaultTermination(
                x_tol=1e-7,
                cv_tol=1e-7,
                f_tol=1e-7,
                nth_gen=50)
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
            print('Running optimization using : ',Model)
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
            print('Optimization completed')
            inputs[0].extend([target_var, op])  #inserting target time period and optimization results 
            colnames = ['NOSt','NOSp','SL', 'OR_opt', 'K_opt', 'T_tgt', 'T_opt']

            df = pd.DataFrame(inputs, columns=colnames,index=['Output']).round(3)

            display(df)
            begin = False
            
    button = widgets.Button(description="Run optimization", button_style = 'success', tooltip = 'Description', icon = 'compact-disc')                    
    output = widgets.Output()

    display(button, output)
    button.on_click(on_button_clicked)
    pass