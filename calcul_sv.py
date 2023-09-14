#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 09:01:06 2022

Flemme de réouvrir des vieux fichiers pour y coller une opération d'aujourd'hui.
Je vais me débrouiller en partant de peu et avec 2 3 copié-collés.

C'est un peu plus long que prévu.
Le calcul des valeurs de Shapley semble encore faire planter ma machine.
Je tente en cbenv1 plutôt que expenv1.

@author: Corentin.Boidot
"""


import numpy as np
import pandas as pd
import shap
import sklearn
import pickle

from brouillons.complementary_functions import df_fx

#### on récupère les 4 fichiers, le modèle et les scores

pd.set_option("display.max_columns",40)
pd.set_option("display.max_rows",30)

path ="data_cache/"

with open(path+"selection_tr.pkl",'rb') as pf: 
    df_train = pickle.load(pf)
# df_select = df_select[new_order]
with open(path+"label_tr.pkl",'rb') as f:
    label_tr = pickle.load(f)

with open(path+"selection_ev.pkl",'rb') as pf: 
    df_eval = pickle.load(pf)
# df_select = df_select[new_order]
with open(path+"label_ev.pkl",'rb') as f:
    label_ev = pickle.load(f)
    # dans quel sens, ce label ?
    # stocké dans Exp, donc 1=victoire des rouges

# punaise, j'ai pas encore entraîné le modèle....
# presque sûr d'avoir pris cette précaution avant l'expe 1.1+

# j'ouvre decoupage2.py et refactor.py en parallèle 
# ( dossier Exp1-Streamlit/brouillons/ )
# re_RF est en fait encore stocké dans these1A...

with open("../these1A/model/"+"re_RF.pkl",'rb') as f:
    rf = pickle.load(f)
# et dans quel sens est-il réglé, je parie qu'il détecte les victoires bleues

with open("../these1A/model/"+"column_transformer.pkl", 'rb') as f:
    c_tr = pickle.load(f)
    # il est "fait pour" garder intact les représentations de blueFirstBlood
    # (représentation de firstBlood de lancien temps)

# restauration des colonnes blueFirstBlood
df_train['firstBlood'] = df_train.firstBlood.apply(lambda x: 0 if x=='red' else 1)
df_eval['firstBlood'] = df_eval.firstBlood.apply(lambda x: 0 if x=='red' else 1)
columns = list(df_train.columns)
columns[0] = "blueFirstBlood"
df_eval.columns, df_train.columns = columns, columns

# un 1er oubli ! les moy

df_train, df_eval = df_train.drop("moy"), df_eval.drop("moy")

# j'espère que ce sera mon dernier oubli... le old_order

old_order = ['blueFirstBlood', 'blueDragons', 'blueHeralds', 'blueTowersDestroyed',
       'redDragons', 'redHeralds', 'redTowersDestroyed', 'blueWardsPlaced',
       'blueWardsDestroyed', 'blueKills', 'blueAssists', 'blueTotalGold',
       'blueTotalExperience', 'blueTotalMinionsKilled',
       'blueTotalJungleMinionsKilled', 'redWardsPlaced', 'redWardsDestroyed',
       'redKills', 'redAssists', 'redTotalGold', 'redTotalExperience',
       'redTotalMinionsKilled', 'redTotalJungleMinionsKilled']

X_tr = df_fx(c_tr.transform,df_train[old_order])
X_ev = df_fx(c_tr.transform,df_eval[old_order])


# je veux prédire la victoire des rouges

pred_tr = df_fx(lambda x: rf.predict_proba(x)[:,0],X_tr)
pred_ev = df_fx(lambda x: rf.predict_proba(x)[:,0],X_ev)


##### remarques faites avant 'old_order'
# hum, je suis bien corrélé au labels (acc à 76.7%)
# accuracy_score(y_true=label_tr, y_pred=pred_tr>.5)
# mais je me rend compte que la Gold ou l'Exp ne nous aident pas (acc<50%)...
# suis-je dans le bon sens ?
#
with open(path+"linear_tr.pkl",'rb') as f:
    lin_tr = pickle.load(f)
#   
# accuracy_score(y_true=label_tr, y_pred=lin_tr>.5) # 73.3%
# on m'aurait dit, depuis le temps...
#
# ça va mieux, le problème était que je ne prenais pas l'original mais X_tr
# pred_naze = (df_train.redKills - df_train.blueKills) >0
# accuracy_score(y_true=label_tr, y_pred=pred_naze) 0.667
# pred_naze = (df_train.redTotalGold - df_train.blueTotalGold) >0
# accuracy_score(y_true=label_tr, y_pred=pred_naze) .7
#####

# accuracy_score(y_true=label_tr, y_pred=pred_tr>.5) #.7667


# Je vais reprendre l'ancien format, ie: les sv sont faites sur le old_order
# mais avec sv_first.
# Mais j'ai bien peur de devoir chercher chez paper1_draft

# réparation, 24 fin de journée
# subtilité : les données pour l'xp°
with open('../these1A/data/lol2_'+"Xtrain.pkl",'rb') as pf:
    df_xp = pickle.load(pf)


xpr = shap.TreeExplainer(rf,df_xp.iloc[:1000])
# attention à préprocesser !
xpo_tr = xpr(X_tr,check_additivity=False)[:,:,1]
#sale bête
xpo_tr.values = -xpo_tr.values
xpo_tr.base_values = 1-xpo_tr.base_values


xpr = shap.TreeExplainer(rf,df_xp)
xpo_ev = xpr(X_ev,check_additivity=False)[:,:,1]
xpo_ev.values = -xpo_ev.values
xpo_ev.base_values = 1-xpo_ev.base_values

df_eval["blueFirstBlood"] = df_eval.blueFirstBlood.apply(
    lambda x: "blue" if x==1 else "red")
df_eval = df_eval.rename(columns={'blueFirstBlood':'firstBlood'})

df_train["blueFirstBlood"] = df_train.blueFirstBlood.apply(
    lambda x: "blue" if x==1 else "red")
df_train = df_train.rename(columns={'blueFirstBlood':'firstBlood'})

xpo_tr.data = df_train
xpo_ev.data = df_eval

with open("shap_tr.pkl",'wb') as pf:
    pickle.dump(xpo_tr,pf)
with open("shap_ev.pkl",'wb') as pf:
    pickle.dump(xpo_ev,pf)
    
# et là normalement, c'est fini, j'ai créé les bons fichiers, en cohérence.

# la seule bizarrerie, c'est à l'index 6935 dans l'eval, un match qui aurait
# clairement dû être gagné par les rouges : le rf donne 100% (et ça se
# comprend vu les données) mais il a dû y avoir un retournement.

# Observons

from brouillons.complementary_functions import error_graph

error_graph(true_labels=label_ev,scores=pred_ev,bins=np.linspace(0,1,11))

error_graph(true_labels=label_tr,scores=pred_tr,bins=np.linspace(0,1,11))
error_graph(true_labels=label_ev[:25],scores=pred_ev[:25],bins=np.linspace(0,1,11))
error_graph(true_labels=label_ev[25:],scores=pred_ev[25:],bins=np.linspace(0,1,11))

# tout a l'air d'aller bien, j'ai de tout partout.

# un truc à faire en plus : enregistrer les preds de rf !

with open("rf_tr.pkl",'wb') as pf:
    pickle.dump(pred_tr,pf)
with open("rf_ev.pkl",'wb') as pf:
    pickle.dump(pred_ev,pf)


### Houston, nous avons un problème : nos SV semblent quasi constantes !

shap.plots.beeswarm(xpo_tr)
# confirmées pathologiques

# calcul plus long le deuxième coup...
# mais toujours patho
# une fois que je change le contenu du xpr(_,check=Flase), ça va mieux.

shap.plots.beeswarm(xpo_ev,max_display=24)


# ok, il me faut des données d'entraînement machine pour l'explicateur !


### 8 juin, je me rends compte que ça ne va pas du tout.
# c'est surtout que les données de l'un et de l'autre ne correspondent pas.




new_order = ['firstBlood', 'blueDragons', 'redDragons', 'blueHeralds',
 'redHeralds', 'blueTowersDestroyed', 'redTowersDestroyed', 'blueWardsPlaced',
 'redWardsPlaced', 'blueWardsDestroyed', 'redWardsDestroyed', 'blueKills', 
 'redKills', 'blueAssists', 'redAssists', 'blueTotalGold', 'redTotalGold',
 'blueTotalExperience', 'redTotalExperience', 'blueTotalMinionsKilled', 
 'redTotalMinionsKilled', 'blueTotalJungleMinionsKilled',
 'redTotalJungleMinionsKilled']

#%% BSHAP style

zeros = 0*df_train.iloc[:2]
zeros = zeros.reset_index(drop=True)
Zeros = df_fx(c_tr.transform,zeros[old_order])

p_0 = df_fx(lambda x: rf.predict_proba(x)[:,0],Zeros)
# 0.315
# dommage, sachant que la pred sur zeros (invalide) m'avait donné du 50%

with open("../these1A/model/"+"re_linear.pkl",'rb') as f:
    lin = pickle.load(f)

df_fx(lambda x: lin.predict_proba(x)[:,0],Zeros)
# 0.35...
# merde

# Je retourne piller déffrichage LoL

#%% Linéaire sans biais

from sklearn import linear_model#, LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

path_d = "../these1A/data/lol2_"

with open(path_d+"Xtrain.pkl",'rb') as pf:
    X_train = pickle.load(pf)
with open(path_d+"Xtest.pkl",'rb') as pf:
    X_test = pickle.load(pf)
with open(path_d+"ytest.pkl",'rb') as pf:
    y_test = pickle.load(pf)
with open(path_d+"ytrain.pkl",'rb') as pf:
    y_train = pickle.load(pf)

X_tr = df_fx(c_tr.fit_transform,X_train)
X_tt = df_fx(c_tr.transform,X_test)

reg = linear_model.LogisticRegression('none',fit_intercept=False)
# reg = linear_model.LogisticRegression('l1',fit_intercept=False,solver='saga')
reg.fit(X_tr,y_train)
X_eval = X_tt#pd.DataFrame(X_tt,columns=X.columns)

ypred = reg.predict(X_eval)
accuracy_score(y_test,ypred)
# 0.7319838056680162

df_fx(lambda x: reg.predict_proba(x)[:,0],Zeros)
# 0.4545
ypred.mean()
# 0.4895
y_train.mean()
#  0.5017

reg.coef_
# Euh... je suis assez surpris des résultats. est-ce à cause  de la normali-
# sation ?

reg.fit(X_train,y_train)
ypred = reg.predict(X_eval)
accuracy_score(y_test,ypred)
# 0.48

df_fx(lambda x: reg.predict_proba(x)[:,0],Zeros)
# 0.500939

#%% Modèle RF sur Lin ?

from sklearn.ensemble import RandomForestClassifier

reg2 = RandomForestClassifier(n_estimators=101,)
reg2.fit(X_tr,y_train)
