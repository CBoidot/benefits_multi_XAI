#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:28:00 2023

On prend découpage2 et...
On arrange.

À éxécuter dans le dossier supérieur.

@author: Corentin.Boidot
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",40)
pd.set_option("display.max_rows",30)

import pickle
from sklearn import linear_model#, LinearRegression, LogisticRegression
from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler, OrdinalEncoder




# with open("sv_vrac_linear.pkl",'rb') as pf:
#     sv_rf = pickle.load(pf)
path_d = '../these1A/data/'
path_m = '../these1A/model/'

with open(path_d+"re_Xtrain.pkl",'rb') as pf:
    X_train = pickle.load(pf)
with open(path_d+"re_Xtest.pkl",'rb') as pf:
    X_test = pickle.load(pf)
    
with open(path_d+"re_ytest.pkl",'rb') as pf:
    # j'ai gardé le stockage originel, où 1 = victoire des bleus
    y_test = 1 - pickle.load(pf)

with open(path_m+"re_linear.pkl",'rb') as pf:
    # j'ai gardé le stockage originel, où 1 = victoire des bleus
    reg = pickle.load(pf)

# X_tt = X_test

ord_c = ['blueFirstBlood',
             'blueDragons',
             'blueHeralds',
             'blueTowersDestroyed',
             'redDragons',
             'redHeralds',
             'redTowersDestroyed',
             'blueWardsPlaced',
             'blueWardsDestroyed',
             'blueKills',
             'blueAssists',
             'blueTotalGold',
             'blueTotalExperience',
             'blueTotalMinionsKilled',
             'blueTotalJungleMinionsKilled',
             'redWardsPlaced',
             'redWardsDestroyed',
             'redKills',
             'redAssists',
             'redTotalGold',
             'redTotalExperience',
             'redTotalMinionsKilled',
             'redTotalJungleMinionsKilled']
categ_c = ord_c[:7]
cont_c = ord_c[7:]
col_transfo = [(name+"_tr",
                OrdinalEncoder(categories=[[0,1]],
                               handle_unknown='use_encoded_value',
                               unknown_value=2),
                [name]) for name in categ_c] + \
                [(name+"_tr", 
                 RobustScaler(quantile_range=(5,95)), 
                 [name]) for name in cont_c]
            
c_tr = ColumnTransformer(col_transfo,
         remainder='passthrough', verbose_feature_names_out=False)

## importer df_fx en même temps que ce qui suit...
# tout est dans complementary_functions.py de Exp1-Streamlit
X_tr = c_tr.fit_transform(X_train)
# tout va bien, on est sous les 10^-16 de différence.

# (9/01) En fait ça sert à rien.

# Bon, en vrai je n'ai qu'à dire qu'il n'y a pas de différence sur les categ.
# Je refit un robust scaler, je le picklise, et lui saura me faire une
# inverse_transform

rs2 =  RobustScaler(quantile_range=(5,95))
rs2.fit(X_train[cont_c])

with open('robust_scaler.pkl',"wb") as f:
    pickle.dump(rs2,f)
