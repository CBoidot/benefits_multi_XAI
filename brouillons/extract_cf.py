#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:54:12 2022

Bon, il me faut une base historique pour mes contrefactuels / prototypes.
Je l'extrais ici.

# 17 jan
Je me décide à regarder les plus proches voisins en norme l1.
J'appelerais mes fichiers 'near1' :)

@author: Corentin.Boidot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",40)
pd.set_option("display.max_rows",30)

import json
import pickle
# from sklearn import linear_model#, LinearRegression, LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.preprocessing import  RobustScaler, OrdinalEncoder


path_d = '../these1A/data/'
path_m = '../these1A/model/'
lpath = 'data_cache/'

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
categ_c = ord_c[:7]
cont_c = ord_c[7:]
# col_transfo = [('categories',
#                 OrdinalEncoder(categories=[[0,1]]*7,
#                                handle_unknown='use_encoded_value',
#                                unknown_value=2),
#                 categ_c),
#                 ('continues',  
#                  RobustScaler(quantile_range=(5,95)), 
#                  cont_c)]
# c_tr = ColumnTransformer(col_transfo,
#          remainder='passthrough', verbose_feature_names_out=False)

## importer df_fx en même temps que ce qui suit...
# tout est dans complementary_functions.py de Exp1-Streamlit
# ref = c_tr.fit_transform(X_train)
with open('../these1A/model/column_transformer.pkl','rb') as f:
    c_tr = pickle.load(f)
ref = c_tr.transform(X_train)

with open('../these1A/model/re_RF.pkl','rb') as f:
    model = pickle.load(f)


with open(lpath+"selection_tr.pkl",'rb') as f:
    X_tr = pickle.load(f)

with open(lpath+"selection_ev.pkl",'rb') as f:
    X_ev = pickle.load(f)

X_tr = X_tr.rename(columns={"firstBlood":"blueFirstBlood"}).drop('moy')
X_ev = X_ev.rename(columns={"firstBlood":"blueFirstBlood"}).drop('moy')

Xtr = c_tr.transform(X_tr[ord_c])
Xev = c_tr.transform(X_ev[ord_c])
# (5jan) j'eus été plus avisé de tout mettre au new order... pas grave.


# HERE
mode = "l2"

d_tr = pairwise_distances(ref,Xtr,metric=mode)
d_ev = pairwise_distances(ref,Xev,metric=mode)

# Bien. Now, avec labels

with open(path_d+"re_ytrain.pkl",'rb') as pf:
    y_train = pickle.load(pf) # 1 == blue # contient pas les suivants
with open(lpath+"label_tr.pkl",'rb') as pf:
    ytr = pickle.load(pf) # 1 == red
with open(lpath+"label_ev.pkl",'rb') as pf:
    yev = pickle.load(pf)

y_ref = y_train

#%% ana

proto_tr_red = euclidean_distances(ref[y_train==0],Xtr[ytr==1])
proto_tr_blue = euclidean_distances(ref[y_train==1],Xtr[ytr==0])
proto_ev_red = euclidean_distances(ref[y_train==0],Xev[yev==1])
proto_ev_blue = euclidean_distances(ref[y_train==1],Xev[yev==0])

# plt.hist([min(proto_ev_red[:,i]) for i in range(25)],bins=20) #  bref
# plt.hist([min(proto_ev_blue[:,i]) for i in range(25)],bins=20)

cf_tr_red = euclidean_distances(ref[y_train==1],Xtr[ytr==1])
cf_tr_blue = euclidean_distances(ref[y_train==0],Xtr[ytr==0])
cf_ev_red = euclidean_distances(ref[y_train==1],Xev[yev==1])
cf_ev_blue = euclidean_distances(ref[y_train==0],Xev[yev==0])

d_tr = pd.DataFrame(d_tr,index=X_train.index,columns=X_tr.index)
d_ev = pd.DataFrame(d_ev,index=X_train.index,columns=X_ev.index)

nn10 = []

for i in range(30):
    # je veux les 10 plus proches, avec leur valences relatives à mon cas
    ind = d_tr.iloc[:,i].sort_values().iloc[:1].index
    # print(d_tr.loc[ind,ytr.index[i]])
    # print(y_train.loc[list(ind)])
    nn10.append((y_train.loc[list(ind)].mean()<=0)==ytr.iloc[i])
    # p,cf,j = -1, -1, 0
    # nearests = d_tr.iloc[:,i].sort_values()
    # while p==-1 or cf==-1:
    #     if y_train.loc[nearests.index[j]]==ytr.iloc[i]:
    #         if cf==-1:
    #             cf = j
    #     else:
    #         if p==-1:
    #             p = j
    #     j += 1
    # print("Prototype at {} and CF at {}.".format(p,cf))

nn10 = []

for i in range(50):
    # je veux les 10 plus proches, avec leur valences relatives à mon cas
    # ind = d_ev.iloc[:,i].sort_values().iloc[:1].index
    # print(d_ev.loc[ind,yev.index[i]])
    # print(y_train.loc[list(ind)])
    # nn10.append((y_train.loc[list(ind)].mean()<=0)==yev.iloc[i])
    p,cf,j = -1, -1, 0
    nearests = d_ev.iloc[:,i].sort_values()
    while p==-1 or cf==-1:
        if y_train.loc[nearests.index[j]]==yev.iloc[i]:
            if cf==-1:
                cf = j
        else:
            if p==-1:
                p = j
        j += 1
    print("Prototype at {} and CF at {}.".format(p,cf))
    
#%% chargement de l'échantillon (train)
subsample = X_train.sample(n=1000)

# with open('reference.pkl','wb') as f: # des fois que ça resserve...
#     pickle.dump(subsample,f)
with open('reference.pkl','rb') as f: # et oui
    subsample = pickle.load(f)

#%% go on sauvegarde les plus proches (tr)


ref = c_tr.fit_transform(subsample)
d_tr = pairwise_distances(ref,Xtr,metric=mode)
d_ev = pairwise_distances(ref,Xev,metric=mode)
d_tr = pd.DataFrame(d_tr,index=subsample.index,columns=X_tr.index)
d_ev = pd.DataFrame(d_ev,index=subsample.index,columns=X_ev.index)



nearest = pd.DataFrame(columns=ord_c)
y = []
# dist = []
for i in range(30):
    # nearest.append(d_tr.iloc[:,i].sort_values().iloc[:1])
    nearest = \
        nearest.append(subsample.loc[d_tr.iloc[:,i].sort_values().index[0]])
    y.append(y_ref.loc[d_tr.iloc[:,i].sort_values().index[0]])

# 27 février : j'intègre enfin la pred du modèle
pred = model.predict_proba(c_tr.transform(nearest))[:,0]

nearest['pred'] = pred
nearest['target'] = 1-np.array(y)

nearest = nearest.rename(columns={"blueFirstBlood":"firstBlood"})
# 1er mars : je m'aperçois que les index sont issus de "subsample", ce sont
# ceux des plus proches voisins et non de leurs références. GO corriger
nearest.index = X_tr.index

# with open(lpath+'near_tr.pkl','wb') as f:
#     pickle.dump(nearest,f)

# with open(lpath+'near1_tr.pkl','rb') as f:
#     nearest_bis= pickle.load(f)

#%%idem (ev)

nearest = pd.DataFrame(columns=ord_c)
y = []
# dist = []
for i in range(50):
    # nearest.append(d_tr.iloc[:,i].sort_values().iloc[:1])
    nearest = \
        nearest.append(subsample.loc[d_ev.iloc[:,i].sort_values().index[0]])
    y.append(y_ref.loc[d_ev.iloc[:,i].sort_values().index[0]])

pred = model.predict_proba(c_tr.transform(nearest))[:,0]

nearest['pred'] = pred
nearest['target'] = 1-np.array(y)

nearest = nearest.rename(columns={"blueFirstBlood":"firstBlood"})
nearest.index = X_ev.index


# with open(lpath+'near_ev.pkl','wb') as f:
#     pickle.dump(nearest,f)

#%% distance de l'alter voisin

nan = [] # nearest alternative neighbor

for i in range(50):
    # je veux les 10 plus proches, avec leur valences relatives à mon cas
    # ind = d_ev.iloc[:,i].sort_values().iloc[:1].index
    # print(d_ev.loc[ind,yev.index[i]])
    # print(y_ref.loc[list(ind)])
    # nn10.append((y_ref.loc[list(ind)].mean()<=0)==yev.iloc[i])
    av, j = -1, 1
    nearests = d_ev.iloc[:,i].sort_values()
    while av==-1:
        if y_ref.loc[nearests.index[j]]!=y_ref.loc[nearests.index[0]]:
            av = j
        j += 1
    nan.append(av)
    # print("Nearest Neighbor at 0 and alternative at {}.".format(av))
nan = pd.Series(nan,index=yev.index)

with open(lpath+'alterNN_ev.pkl','wb') as f:
    pickle.dump(nan,f) # j'ai hésité avec list/json, mais l'index sera utile

nan_tr = [] # nearest alternative neighbor

for i in range(30):
    # je veux les 10 plus proches, avec leur valences relatives à mon cas
    # ind = d_ev.iloc[:,i].sort_values().iloc[:1].index
    # print(d_ev.loc[ind,yev.index[i]])
    # print(y_ref.loc[list(ind)])
    # nn10.append((y_ref.loc[list(ind)].mean()<=0)==yev.iloc[i])
    av, j = -1, 1
    nearests = d_tr.iloc[:,i].sort_values()
    while av==-1:
        if y_ref.loc[nearests.index[j]]!=y_ref.loc[nearests.index[0]]:
            av = j
        j += 1
    nan_tr.append(av)
    # print("Nearest Neighbor at 0 and alternative at {}.".format(av))

nan_tr = pd.Series(nan_tr,index=ytr.index)

with open(lpath+'alterNN_tr.pkl','wb') as f:
    pickle.dump(nan_tr,f)

    
#%% CF pris dans le test

with open(path_d+"re_Xtest.pkl",'rb') as pf:
    X_test = pickle.load(pf)

X_test.index.duplicated().any()
# Out[98]: False # j'ai testé également les intersections avec nos 2 amis

index = X_test.index.difference(X_tr.index).difference(X_ev.index)
index.shape[0]+80
# Out[111]: 2423

subsample = X_test.loc[index].sample(n=1000)

# bah... je renomme les anciennnes versions et j'applique le code au dessus

with open(path_d+"re_ytest.pkl",'rb') as f:
    y_ref = pickle.load(f).loc[index]
