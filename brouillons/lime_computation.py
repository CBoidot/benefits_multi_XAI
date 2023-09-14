#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:42:33 2023

@author: Corentin.Boidot
"""

import numpy as np
import pandas as pd

import re
import sklearn
import sklearn.datasets
import sklearn.ensemble

import lime
from lime import lime_tabular
# from __future__ import print_function
from shap import Explanation

import pickle

# np.random.seed(1)

path_m = '../these1A/model/'
path_d = '../these1A/data/'
path = 'data_cache/'

with open(path_m+"re_RF.pkl",'rb') as f:
    rf = pickle.load(f)


with open(path_d+"re_Xtrain.pkl",'rb') as pf:
    X_train = pickle.load(pf)
    

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
ord_cplusplus = ['firstBlood']+ord_c[1:]

with open('../these1A/model/column_transformer.pkl','rb') as f:
    c_tr = pickle.load(f)
ref = c_tr.transform(X_train)


with open("selection_tr.pkl",'rb') as f:
    X_tr = pickle.load(f)

with open("selection_ev.pkl",'rb') as f:
    X_ev = pickle.load(f)

X_tr = X_tr.rename(columns={"firstBlood":"blueFirstBlood"}).drop('moy')
X_ev = X_ev.rename(columns={"firstBlood":"blueFirstBlood"}).drop('moy')

Xtr = c_tr.transform(X_tr[ord_c])
Xev = c_tr.transform(X_ev[ord_c])

with open("label_ev.pkl",'rb') as f:
    labels_ev = pickle.load(f)
with open("label_tr.pkl",'rb') as f:
    labels_tr = pickle.load(f)

sklearn.metrics.accuracy_score(labels_ev, rf.predict(Xev))
with open("selection_tr.pkl",'rb') as f:
    X_tr = pickle.load(f).drop('moy')

with open("selection_ev.pkl",'rb') as f:
    X_ev = pickle.load(f).drop('moy')

#%% calculs lime

explainer = lime_tabular.LimeTabularExplainer(ref,
            feature_names=ord_c, class_names=['red','blue'])#,
            # discretize_continuous=True)

### je fais des visu
# i = np.random.randint(0, Xev.shape[0])
# exp = explainer.explain_instance(Xev[i], rf.predict_proba, num_features=23, top_labels=1)


xp_list = []
intercepts = []
for i in range(Xtr.shape[0]):
    xp = explainer.explain_instance(Xtr[i], rf.predict_proba, num_features=23,
                                     top_labels=1)
    ft_order = []
    team = rf.predict(Xtr[i:i+1])[0]
    intercept = xp.intercept
    xp = xp.as_list(label=team)
    for line in xp:
        name = [x for x in re.findall('[A-Za-z]*',line[0]) if len(x)>0][0]
        ft_order.append(name)
    if team==0: #'red'
        # [1] car on récupère la FI seulement
        xp = pd.Series([x[1] for x in xp],index=ft_order)[ord_c]
        intercepts.append(intercept[team])
    elif team==1: #'blue'
        xp = pd.Series([-x[1] for x in xp],index=ft_order)[ord_c]
        intercepts.append(1-intercept[team])
    xp_list.append(xp.values)

# from lime.discretize import QuartileDiscretizer
    
lime_tr = np.array(xp_list)
# /!\ je me suis fais avoir une fois par mes manip' d'X_tr, pas deux.
X_tr = X_tr.rename(columns={"blueFirstBlood":"firstBlood"})
e = Explanation(lime_tr,base_values=np.array(intercepts),data=X_tr,
                feature_names=ord_cplusplus)
#base_values=np.zeros((30,1))

with open(path+"lime_tr.pkl","wb") as f:
    pickle.dump(e,f) #hmm... méfiance, l'index à l'air trié.
    
xp_list = []
intercepts = []
for i in range(Xev.shape[0]):
    xp = explainer.explain_instance(Xev[i], rf.predict_proba, num_features=23,
                                     top_labels=1)
    ft_order = []
    team = rf.predict(Xev[i:i+1])[0]
    intercept = xp.intercept
    xp = xp.as_list(label=team)
    for line in xp:
        name = [x for x in re.findall('[A-Za-z]*',line[0]) if len(x)>0][0]
        ft_order.append(name)
    if team==0: #'red'
        # [1] car on récupère la FI seulement
        xp = pd.Series([x[1] for x in xp],index=ft_order)[ord_c]
        intercepts.append(intercept[team])
    elif team==1: #'blue'
        xp = pd.Series([-x[1] for x in xp],index=ft_order)[ord_c]
        intercepts.append(1-intercept[team])   
    xp_list.append(xp.values) 
    
lime_ev = np.array(xp_list)
X_ev = X_ev.rename(columns={"blueFirstBlood":"firstBlood"})
e = Explanation(lime_ev,base_values=np.array(intercepts),data=X_ev,
                feature_names=ord_cplusplus)

with open(path+"lime_ev.pkl","wb") as f:
    pickle.dump(e,f) 


#%% divagation

import shap
import matplotlib.pyplot as plt

shap.plots.waterfall(e[2],show=False) # ax=
fig = plt.gcf()
# plt.xlim([.1, .9])
# # axes=ax.gca()
axes,_, _ = fig.get_axes()
axes.set_xlim([.4,.6])
# fig.show()

# bon, ça ne donne pas un résultat fabuleux...

#%% Stabilité de LIME ?




