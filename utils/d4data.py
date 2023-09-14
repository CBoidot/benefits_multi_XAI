#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:18:29 2023

J'ai testé: on peut faire du cash depuis un fichier externe, une coup je 
localise ma préparation des données ici.

@author: Corentin.Boidot
"""

import numpy as np
import streamlit as st

import pickle


path = ""
path = "data_cache/"
pathr = 'results2/'


old_order = ['firstBlood', 'blueDragons', 'blueHeralds', 'blueTowersDestroyed',
       'redDragons', 'redHeralds', 'redTowersDestroyed', 'blueWardsPlaced',
       'blueWardsDestroyed', 'blueKills', 'blueAssists', 'blueTotalGold',
       'blueTotalExperience', 'blueTotalMinionsKilled',
       'blueTotalJungleMinionsKilled', 'redWardsPlaced', 'redWardsDestroyed',
       'redKills', 'redAssists', 'redTotalGold', 'redTotalExperience',
       'redTotalMinionsKilled', 'redTotalJungleMinionsKilled']
new_order = ['firstBlood', 'blueDragons', 'redDragons', 'blueHeralds',
 'redHeralds', 'blueTowersDestroyed', 'redTowersDestroyed', 'blueWardsPlaced',
 'redWardsPlaced', 'blueWardsDestroyed', 'redWardsDestroyed', 'blueKills', 
 'redKills', 'blueAssists', 'redAssists', 'blueTotalGold', 'redTotalGold',
 'blueTotalExperience', 'redTotalExperience', 'blueTotalMinionsKilled', 
 'redTotalMinionsKilled', 'blueTotalJungleMinionsKilled',
 'redTotalJungleMinionsKilled']


# @st.cache # 1/08/23-> je déplacce intelligemment ce décorateur dans main2.
def init_data_tr(seed,ntr):
    '''
    Returns
    df_select : pd.DataFrame, passé au new_order
    '''
    with open(path+"selection_tr.pkl",'rb') as pf: 
        df_select = pickle.load(pf)
    df_select = df_select[new_order]
    with open(path+"label_tr.pkl",'rb') as f:
        labels = pickle.load(f)
        # 1 = victoire des rouges (ne pas se fier au nom de colonne)
    ### Préprocessing
    # on va randomiser n fonction de l'utilisateur
    # mais chaque bloc de 5 doit contenir au moins 1 rouge et un bleu.
    final_index = []
    blue_draw = []
    red_draw = []
    penury = 0 # 1 = rouge, -1 = bleu
    
    labelshuffled = labels.sample(frac=1,random_state=seed)
    # phase de tri : chaque bloc de 5 doit avoir au moins 1 rouge et 1 bleu
    i = 0
    while len(final_index)<ntr:
        cur_indx = labelshuffled.index[i]
        if penury==0: # cas normaux
            if len(final_index)%5 != 4:
                final_index.append(cur_indx) # inser
                if labelshuffled.iloc[i]:
                    red_draw.append(cur_indx)
                else:
                    blue_draw.append(cur_indx)
            else: # à la fin d'un paquet de 5, on contrôle, et on déclenche la
                # pénurie si nécessaire
                if blue_draw==[] :
                    penury = -1
                    red_draw = [cur_indx]
                elif red_draw==[]:
                    penury = 1
                    blue_draw = [cur_indx]
                else:
                    final_index.append(cur_indx) #inser
                    blue_draw = []
                    red_draw = []
                    # print("bloc ok")
        elif penury==1: # on manque de rouge
            if not labelshuffled.iloc[i]:
                blue_draw.append(cur_indx) # on empile les bleus en trop
            else:
                final_index.append(cur_indx) # inser
                # le précédent bloc est donc fini, gestion de la pile now
                if len(blue_draw)<5: # fin de crise
                    final_index += blue_draw # multi-inser
                    penury = 0
                else: # on prépare le prochain bloc qui attendra son rouge
                    final_index += blue_draw[:4] # multi-inser
                    blue_draw = blue_draw[4:] # la pile est réduite
        elif penury==-1: # on manque de bleu
            if labelshuffled.iloc[i]:
                red_draw.append(cur_indx) # on empile les bleus en trop
            else:
                final_index.append(cur_indx) # inser
                # le précédent bloc est donc fini, gestion de la pile now
                if len(red_draw)<5: # fin de crise
                    final_index += red_draw # multi-inser
                    penury = 0
                else: # on prépare le prochain bloc qui attendra son bleu
                    final_index += red_draw[:4] # multi-inser
                    red_draw = red_draw[4:] # réduction de la pile
        i+=1
    labels = labels.loc[final_index[:ntr]]
    m = df_select.loc["moy"]
    df_select = df_select.loc[final_index]
    df_select.loc["moy"] = m
    
    with open(pathr+"shuffle"+str(seed)+".pkl",'wb') as f:
        pickle.dump(labels,f)
    
    labels = labels.reset_index(drop=True)
    # sans quoi, je ne peux les comparer pour donner une note à l'utilisateur
    # print("Load train")
    return df_select, labels

# @st.cache
def init_data_ev():
    '''
    df_select : pd.DataFrame, passé au new_order
    '''
    # N = 40 # index à partir duquel on insère 
    with open(path+"selection_ev.pkl",'rb') as pf: 
        df_select = pickle.load(pf)
    df_select = df_select[new_order]
# ### traitement de la répétition pour évaluation du bruit de décision
#     # en maquette, je vais les répéter dans la deuxième dizaine.
#     to_insert = df_select.iloc[:10].sample(5,random_state=USER)
#     with open(pathr+"redondance_"+str(USER)+".pkl",'wb') as f:
#         # pickle.dump(to_insert,f)
#     df_bis = df_select.iloc[:N].reset_index(drop=True)
#     for i in range(5):
#         df_bis.loc[N+2*i] = df_select.iloc[10+i] 
#         df_bis.loc[N+2*i+1] = to_insert.iloc[i]
#     df_bis = pd.concat([df_bis,df_select.reset_index(drop=True).iloc[N+5:50]]) \
#         .reset_index(drop=True)
#     df_bis.loc['moy'] = df_select.loc['moy']
    return df_select

@st.cache
def init_fi(ntr,ref,mode="shap"):
    '''
    xpo_tr : explication telle que définies dans shap. xpo.data est en old order
    xpo_ev : explication telle que définies dans shap. xpo.data est en old order
    pred_tr : np.array
    pred_ev : np.array
    '''
    with open(path+mode+"_tr.pkl",'rb') as pf:
        xpo_tr = pickle.load(pf)
    with open(path+"rf_tr.pkl",'rb') as pf:
        pred_tr = pickle.load(pf).values
    with open(path+mode+"_ev.pkl",'rb') as pf:
        xpo_ev = pickle.load(pf)
    with open(path+"rf_ev.pkl",'rb') as pf:
        pred_ev = pickle.load(pf).values
    # correction
    xpo_tr.data = xpo_tr.data[old_order]
    xpo_ev.data = xpo_ev.data[old_order]
    
    # order !!
    # un moyen d'avoir la permutation
    perm = [xpo_tr.data.index.get_indexer_for([ref.index[i]])[0] for i in range(ntr)]
    # print(perm)
    xpo_tr = xpo_tr[perm]
    # xpo_tr[perm].data.index==df_tr.index.drop("moy") # True!
    pred_tr = pred_tr[perm]
    
    return xpo_tr, xpo_ev, pred_tr, pred_ev

@st.cache
def init_likelihood():
    with open(path+"likeli_tr.pkl",'rb') as pf:
        likeli_tr = pickle.load(pf)
    with open(path+"likeli_ev.pkl",'rb') as pf:
        likeli_ev = pickle.load(pf)
    return likeli_tr,likeli_ev

@st.cache
def init_simil(mode=2,alter=False):
    if mode==2:
        with open(path+"near_tr.pkl",'rb') as pf:
            near_tr = pickle.load(pf)
        with open(path+"near_ev.pkl",'rb') as pf:
            near_ev = pickle.load(pf)
    elif mode==1:
        with open(path+"near1_tr.pkl",'rb') as pf:
            near_tr = pickle.load(pf)
        with open(path+"near1_ev.pkl",'rb') as pf:
            near_ev = pickle.load(pf)
    cols = new_order+['pred','target']
    if alter:
        if mode==1:
            with open(path+"alterNN1_tr.pkl",'rb') as f:
                ann_tr = pickle.load(f)
            with open(path+"alterNN1_ev.pkl",'rb') as f:
                ann_ev = pickle.load(f)
        elif mode==2:
            with open(path+"alterNN_tr.pkl",'rb') as f:
                ann_tr = pickle.load(f)
            with open(path+"alterNN_ev.pkl",'rb') as f:
                ann_ev = pickle.load(f)
        else:
            raise Exception('You have to compute alternative nearest neighbor (change mode to 1)')
        return (near_tr[cols],near_ev[cols], 
                ann_tr,ann_ev)
    else:
        return near_tr[cols],near_ev[cols]