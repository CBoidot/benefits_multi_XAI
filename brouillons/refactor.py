#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:00:31 2022
exécuté avec cbenv1...


Éxécutions uniques, par blocs, mis en commentaire après éxécution.

beaucoup de copiés-collés de deffritchage_lol2

Le but est de réentraîner tout avec les nouvelles données (on a virés les 
outliers des wards), et de rééchantilloner (à l'échelle d'un train-test 'IA',
l'échantillonage pour l'expérience humaine est fait ailleurs).

C'est ici qu'on trouve de quoi récupérer le préprocessing.

@author: Corentin.Boidot
"""

import pandas as pd

import pickle

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import RobustScaler, OrdinalEncoder#, LabelEncoder

from utils.complementary_functions import df_f, df_fx #execute depuis these1A


path_d = "../these1A/data/"
path_m = "../these1A/model/"


pd.set_option("display.max_columns",40)
pd.set_option("display.max_rows",30)



with open(path_d+"lol2_"+"Xtrain.pkl",'rb') as pf:
    X_train = pickle.load(pf)
with open(path_d+"lol2_"+"Xtest.pkl",'rb') as pf:
    X_test = pickle.load(pf)
with open(path_d+"lol2_"+"ytest.pkl",'rb') as pf:
    y_test = pickle.load(pf)
with open(path_d+"lol2_"+"ytrain.pkl",'rb') as pf:
    y_train = pickle.load(pf)
    

# # remove outliers    
# X_train = X_train[(X_train.blueWardsPlaced<100)]
# X_test = X_test[(X_test.blueWardsPlaced<100)]
# X_train = X_train[(X_train.redWardsPlaced<100)]
# X_test = X_test[(X_test.redWardsPlaced<100)]
# # X_train de 7409 à 7239 lignes
# # X_test de 2470 à 2423 lignes

# y_train = y_train[(X_train.blueWardsPlaced<100)&(X_train.redWardsPlaced<100)]
# # de 0.501687137265488 à 0.5022793203481144 en moyenne
# y_test = y_test[(X_test.blueWardsPlaced<100)&(X_test.redWardsPlaced<100)]
# # 0.4910931174089069 à 0.4915394139496492 en moyenne

# categ_c = ['blueFirstBlood','blueDragons','blueHeralds','blueTowersDestroyed',
#             'redDragons','redHeralds','redTowersDestroyed']
# X_cont = X_train.drop(categ_c,axis=1)
# cont_c = X_cont.columns

# c_tr = ColumnTransformer(
#           [('categories',
#           OrdinalEncoder(categories=[[0,1]]*7,
#                           handle_unknown='use_encoded_value',
#                           unknown_value=2),
#           categ_c),
#           ('continues',  # je passe à du .95 plutôt que .75 sinon WardsPlaced
#             RobustScaler(quantile_range=(5,95)), # ... pose problème 
#             cont_c)],
#           remainder='passthrough', verbose_feature_names_out=False)

# X_tr = df_fx(c_tr.fit_transform,X_train)

# with open(path_m+"column_transformer", 'wb') as f:
#     pickle.dump(c_tr,f)

# X_tt = df_fx(c_tr.transform,X_test)

# # entraînement régression logistique
# reg = LogisticRegression()
# reg.fit(X_tr,y_train)
# # acc 0.7346264960792406

# reg2 = RandomForestClassifier(n_estimators=200)
# cv = GridSearchCV(reg2, scoring=make_scorer(accuracy_score),
#      cv=10,param_grid={})#KFold(n_splits=10, shuffle=True),n_iter=30)

# cv.fit(X_tr,y_train)
# #best_estimator_ 0.7238959966983078

# with open(path_m+"re_linear.pkl",'wb') as pf:
#     pickle.dump(reg,pf)

# with open(path_m+"re_RF.pkl",'wb') as pf:
#     pickle.dump(cv.best_estimator_,pf)

# with open(path_d+"re_"+"Xtrain.pkl",'wb') as pf:
#     pickle.dump(X_train,pf)
# with open(path_d+"re_"+"Xtest.pkl",'wb') as pf:
#     pickle.dump(X_test,pf)
# with open(path_d+"re_"+"ytest.pkl",'wb') as pf:
#     pickle.dump(y_test,pf)
# with open(path_d+"re_"+"ytrain.pkl",'wb') as pf:
#     pickle.dump(y_train,pf)