#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:59:56 2022

Les fonctions utilisée dans main_f

Je commence le 3 janvier, pour travailler mes démos.
Je mets ici des fonctions d'affichage.

Les datas et potentielles pages devraient être mises ailleurs.

@author: Corentin.Boidot
"""


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import shap
import time
from functools import reduce

from copy import deepcopy
from utils.translate_rule import translate_rule

path = "data_cache/"

index = ['firstBlood', 'Dragons', 'Heralds',
       'TowersDestroyed', 'WardsPlaced', 'WardsDestroyed',
       'Kills', 'Assists', 'TotalGold', 'TotalExperience',
       'TotalMinionsKilled','TotalJungleMinionsKilled']

arg_priority = ['TotalGold','TotalExperience','Kills','Assists',
'WardsPlaced','TotalJungleMinionsKilled','TotalMinionsKilled','WardsDestroyed',
'Dragons','firstBlood','Heralds','TowersDestroyed']

desc_format = {'firstBlood': "Le premier kill a été réalisé par l'équipe {}.",
       'blueDragons':"L'équipe {} a vaincu {} dragons.", 
       'blueHeralds': "L'équipe {} a vaincu {} héraults.",
       'blueTowersDestroyed': "L'équipe {} a détruit {} tourelles.",
       'blueWardsPlaced': "L'équipe {} a placé {} balises.",
       'blueWardsDestroyed': "L'équipe {} a détruit {} balises.",
       'blueKills': "L'équipe {} a tué {} champions ennemis.",
       'blueAssists': "L'équipe {} a fait {} assists.",
       'blueTotalGold':"L'équipe {} a gagné {} pièces d'or.",
       'blueTotalExperience':"L'équipe {} a gagné {} points d'expérience.",
       'blueTotalMinionsKilled': "L'équipe {} a tué {} sbires ennemis.",
       'blueTotalJungleMinionsKilled':"L'équipe {} a tué {} monstres de la jungles."}

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

nondutout = ["Non pas du tout", "non", "plutôt non", "indécis", "plutôt oui", "oui", "oui tout à fait"]
pasdaccord = ["Pas du tout d’accord", "pas d’accord", "plutôt pas d’accord", "neutre", "plutôt d’accord", "d’accord", "tout à fait d’accord"]

options_ope = ["l’équipe bleue va certainement gagner",
            "l’équipe bleue va vraisemblablement gagner",
            "l’équipe bleue a un léger avantage sur l’équipe rouge",
            "je ne sais pas",
            "l’équipe rouge a un léger avantage sur l’équipe bleue",
            "l’équipe rouge va vraisemblablement gagner",
            "l’équipe rouge va certainement gagner"]

#%% displays

def display_match(i,df,col=st,show_mean=True):
    """attention, les i que vous entrez commencent à 1"""
    # print("number: {}".format(i))
    if df is None:
        # if st.session_state.train:
        #     df = df_tr
        # else:
        #     df = df_ev
        raise Exception("Missing dataset")
        # plutôt que de modifier beaucoup de code et écrire des exceptions,
        # je vire les valeurs par défauts (les "df=None," deviennent "df,")
    
    if not show_mean: # oneliner version, utilisée précédemment
        tab = df.iloc[[i]].T
        tv = tab.values
        new_v = np.concatenate([tv[1::2,0:1],tv[2::2,0:1]],axis=1)#.astype(int)
        if tab.iloc[0,0] == 'red':
            new_df = pd.DataFrame([[0,1]],columns=['blue','red'])
        else: # 'blue'
            new_df = pd.DataFrame([[1,0]],columns=['blue','red'])
        new_df = pd.concat([new_df,pd.DataFrame(new_v,columns=['blue','red'])])
    else:
        # if i==-1: # 21 mars 2023
        #     i = -2
        # print(i)
        tab = df.iloc[[i,-1]].T
        tv = tab.values
        double_colonne = np.concatenate([tv[1::2,0:1],tv[2::2,0:1]],axis=1)#.astype(int)
        m_col = np.mean([tv[2::2,1:2],tv[1::2,1:2]],axis=0).astype(float).round(2)
        # if df.equals(df_tr):
        #     # print(tv)
        #     print(double_colonne)
        new_v = np.concatenate((double_colonne,m_col),axis=1)
        if tab.iloc[0,0] == 'red':
            new_df = pd.DataFrame([[0,1,.5]],columns=['blue','red','mean'])
        else: # 'blue'
            new_df = pd.DataFrame([[1,0,.5]],columns=['blue','red','mean'])
        new_df = pd.concat([new_df,
                            pd.DataFrame(new_v,columns=['blue','red','mean'])
                            ],axis=0)

    new_df.index = index
    if col is None:
        new_df.astype(str)
    else:
        col.table(new_df.astype(str))
            
        # tab = df.iloc[[i,-1]].T
        # tv = tab.copy()
        # tv.iloc[:,0] = 0
        # tv = tv.iloc[[i]].T.astype(int).values
        # double_colonne = np.concatenate([tv[1::2,0:1],tv[2::2,0:1]],axis=1)#.astype(int)
        # tv = tab.values
        # m_col = np.mean([tv[2::2,1:2],tv[1::2,1:2]],axis=0).astype(float).round(2)

def display_score(i,pred,col=st,wording=1,hide_score=False,title=False):
    
    # if pred is None:
    #     if st.session_state.train:
    #         pred = pred_tr
    #     else:
    #         pred = pred_ev
    txt = "" if title else "##"
    if wording == 1:
        txt = txt + "# Victoire prédite : équipe _{}_"
    elif wording == 2:
        txt = txt + "# Pronostic IA : équipe _{}_"
    if not hide_score:
        txt += "\n #### Probabilité estimée : _{}_%"
    if pred[i-1]>.5:
        team, percentage = "rouge", str(int(pred[i-1]*100))
    else:
        team, percentage = 'bleue', str(int((1-pred[i-1])*100))
    col.markdown(txt.format(team,percentage))

def display_shap(i,xpo,col=st,plot='bar'):#,pred
    
    # if xpo is None:
    #     if st.session_state.train:
    #         xpo = xpo_tr
    #     else:
    #         xpo = xpo_ev
    if plot == 'bar':
        shap.plots.bar(xpo[i-1],show_data=True,show=False)#st.session_state.cur_match
    elif plot == 'flot':
        e = deepcopy(xpo[i-1])
        # e.base_values = .5
        shap.plots.waterfall(e,max_display=7,show=False)
    elif plot == 'line': # moche
        shap.plots.decision(.5,xpo[i-1].values,xpo[i-1].data,show=False,)
        #feature_display_range=slice(-6,None)
    fig = plt.gcf()
    col.pyplot(fig)
    
def display_likelihood(i,lk,col=st):
    th1,th2 = -69, -78
    # if lk is None:
    #     if st.session_state.train:
    #         lk = likeli_tr
    #     else:
    #         lk = likeli_ev
    if lk[i-1] > th1:
        txt = "Données standard."
    elif lk[i-1] <= th2:
        txt = "/!\ Des données avec de telles valeurs n'étaient pas présentes \
            lors de la formation de l'IA."
    else:
        txt = "Données relativement surprenantes pour l'IA."
    col.markdown("### "+txt)

def display_arg_(i,xpo,pred,col=st):
#     """je laisse tombe shap... données brutes"""    
#     if pred is None:
#         if st.session_state.train:
#             pred = pred_tr
#         else:
#             pred = pred_ev
#     if pred[i-1]>.5:
#         team, percentage = "rouge", str(int(pred[i-1]*100))
#     else:
#         team, percentage = 'bleue', str(int((1-pred[i-1])*100))
        
#     if xpo is None:
#         if st.session_state.train:
#             xpo = xpo_tr
#         else:
#             xpo = xpo_ev
            
#     shap_per_ft = xpo.values[[i]].T
    
#     shap_per_pair = shap_per_ft[1::2,0:1] + shap_per_ft[2::2,0:1]
#     shap_per_pair = np.concatenate([shap_per_ft[0:1,0:1],
#                                     shap_per_pair],axis=0)
#     spp_df = pd.DataFrame(shap_per_pair,index=index)
    
#     ready = spp_df[arg_priority]
    
#     col.write(spp_df)
#     # t1 = "{}={} est plus importante que {} ({}) ".format(
#     #     n1,val1,n1bis,val1bis)
    
#     # t2 = "{}={} est plus importante que {} ({}), ".format(
#     #     n2,val2,n2bis,val2bis)
    pass
    
def display_arg(i,data,pred,col=st):
    
    # if pred is None:
    #     if st.session_state.train:
    #         pred = pred_tr
    #     else:
    #         pred = pred_ev
        # raise Exception('Missing pred')
    if pred[i-1]>.5:
        team, percentage = "rouge", str(int(pred[i-1]*100))
    else:
        team, percentage = 'bleue', str(int((1-pred[i-1])*100))
            
    # if data is None:
    #     if st.session_state.train:
    #         data = df_tr
    #     else:
    #         data = df_ev
        # raise Exception('Missing data')
    tab = data.drop('moy').iloc[[i-1]].T
    tv = tab.values
    # print(tv[1::2,0:1])
    blu_ms_red = tv[1::2,0:1] - tv[2::2,0:1]
    
    if tab.iloc[0,0] == 'red':
        new_df = pd.DataFrame([[0]])
    else: # 'blue'
        new_df = pd.DataFrame([[1]])
    new_df = pd.concat([new_df,pd.DataFrame(blu_ms_red)],axis=0)
    new_df.index = index
    # if col is None:
    #     new_df.astype(str)
    # else:
    #     col.table(new_df.astype(str))
    ready = new_df.loc[arg_priority]
    if team == "rouge":
        ready = - ready
    main_arg = None
    counter = None
    second_arg = None
    ft_i = 0
    while main_arg is None or (counter is None and second_arg is None):
        if ready.values[ft_i] > 0:
            if main_arg is None:
                main_arg = arg_priority[ft_i]
            elif second_arg is None:
                second_arg = arg_priority[ft_i]
        elif counter is None:
            counter = arg_priority[ft_i]
        ft_i += 1
    if counter is None:
        txt = "L'équipe {} va gagner car elle a l'avantage en {} et en {}."
        txt = txt.format(team,main_arg,second_arg)
    else:
        txt = "L'équipe {} devrait gagner malgré son retard en {}, au vu de {}."
        txt = txt.format(team,counter,main_arg)
    col.markdown("### "+txt)

def feature_selection_shap(i,data,fi,n=2):
    """
    IMPERATIF: la data doit être au old_order.
    Aucun ordre nouveau ne sera toléré.
    Parameters
    ----------
    i : entier indexant la donnée courante (index 1:n)
    data : pd.DataFrame -> l'ensemble des données.
    fi : pd.DataFrame -> les features importances pour la décision à prendre
    n: int -> nombre de features en sortie.
    Returns
    -------
    res: dictionnaire dont les clés sont les noms de feature, et les valeurs
        sont des tuples (feature_importance, valeur_feature)
    """
    # if data is None:
    #     if st.session_state.train:
    #         data = df_tr
    #     else:
    #         data = df_ev
    # if fi is None:
    #     if st.session_state.train:
    #         xpo = xpo_tr
    #     else:
    #         xpo = xpo_ev
        # xpdf = xpo.data[new_order] #pd.DataFrame(, columns=xpo.data.columns)
        # print("test consistence")
        # print(xpdf.iloc[i-1]==data.iloc[i-1])
    #     fi = pd.DataFrame(xpo.values, columns=xpo.data.columns)
    fi = fi.iloc[i-1]
    abs_fi = fi.apply(abs)
    x = data.iloc[i-1]
    res = dict(abs_fi.T.sort_values(ascending=False).iloc[:n])
    for k,_ in res.items():
        res[k] = (fi[k],x[k])
    return res

# feature_selection = feature_selection_shap

def display_pastilles(i,xpo,col=st,n=2):
    """Copié-collé de iso, date de l'été 2022.
    Je ne me porte garant de rien."""
    if xpo is not None:
        fi_values = feature_selection_shap(i,xpo.data,
                   pd.DataFrame(xpo.values,columns=xpo.data.columns),n=n)
    else:
        fi_values = feature_selection_shap(i,n=n)
    # descr = ["Le professeur Chen joue dans l'équipe rouge",
    #          "l'équipe bleue a eu des croissants au petit déjeuner"]
    # colors = ['r','b']
    colors = []
    descr = []
    # f_names = ["redTotalExperience","firstBlood"]
    # values = [1,0]
    # feat_imps = [1,-1]
    f_names = list(fi_values.keys())
    feat_imps = list(dict(fi_values.values()).keys())
    values = list(dict(fi_values.values()).values())
    if len(values)<n or len(feat_imps)<n:
        raise "il va falloir laisser tomber l'usage de dictionnaire ci-dessus"
    for ii in range(n):    
        feat_name = f_names[ii]
        value_desc = values[ii]
        fi = feat_imps[ii]
        # factorisable ?
        if feat_name[:-3] == 'ood': # pour Blood
            team = 'rouge' if value_desc==1 else 0
            descr.append(desc_format[feat_name].format(team))
        else:
            team_desc = "bleue"
            if feat_name[0:3] == 'red':
                team_desc = 'rouge'
                feat_name = 'blue' + feat_name[3:]
            descr.append(desc_format[feat_name].format(team_desc,value_desc))
        # ?fin factorisable
        if fi>0:
            xp_sense = "r"
        else:
            xp_sense = "b"
        colors.append(xp_sense)
    positions = list(zip([0]*n,list(range(0,-n,-1))))
    fig, ax = plt.subplots()
    for i,xy in enumerate(positions):
        c = plt.Circle(xy,radius=.4,color=colors[i])
        ax.add_patch(c)
        x,y = xy
        plt.text(x+.7, y-.1, descr[i])
    ax.set_aspect("equal")
    ax.axis('off')
    ax.set_xlim([-.5,10])
    ax.set_ylim([-n+.5,.5])
    col.pyplot(fig)
    
def display_nearest(i,xpo,m=None,ann=None,col=st):
    """
    Créé le 21 décembre 22.
    Parameters
    ----------
    i : int, numéro du match
    xpo : DataFrame contenant les plus proches voisins, et leur label
    col : colonne ou 
    """
    # if xpo is None:
    #     if st.session_state.train:
    #         xpo = near_tr
    #     else:
    #         xpo = near_ev
    # print(xpo)
    # print(near_tr)
    lab = xpo['target']
    simi = xpo.drop('target',axis=1) #).append(m)[new_order]
    txt = "Le match le plus similaire dont je me rappelle est celui-ci. "
    
    # col.write(txt+"L'équipe {} avait gagné ce match.".format(team))
    pred = xpo['pred'].values
    if pred[i-1]>.5:
        team_, percentage = "rouge", str(int(pred[i-1]*100))
    else:
        team_, percentage = 'bleue', str(int((1-pred[i-1])*100))
    txt = txt + " J'estimais que l'équipe {} gagnerait, \
        avec une confiance de {}%. ".format(team_,percentage)
    xpo = xpo.drop('pred',axis=1)
    simi = xpo[new_order]
    
    team = 'rouge' if lab.iloc[i-1] else 'bleue'
    txt += "L'équipe {} a finalement gagné ce match.".format(team)
    
    if m is not None:
        display_match(i-1,simi,col,show_mean=True)
    else:
        display_match(i-1,simi,col,show_mean=False)
    if ann is not None:
        team = 'rouges' if not lab.iloc[i-1] else 'bleus'
        txt = txt + "\nSi je cherche un match gagné par les {}, on doit aller \
            chercher au moins le {}e plus proche."
        # col.write(txt.format(team,ann.iloc[i]+1))
        txt = txt.format(team,ann.iloc[i]+1)
    col.markdown("**"+txt+"**")
        # "+1" car le premier est numéroté "0" =D
    
    
def display_rule(i,data,pred,col=st,w=1,fake=False):
    """
    Parameters
    ----------
    i : int, index du match
    data : pd.DataFrame
    pred : array des scores ML
    col : le conteneur streamlit pour votre affichage
    w : wording
    1 --> Si A et B alors l'équipe x gagnera
    2 --> Comme A et B, l'équipe x devrait gagner.
    """
    with open(path+"rules_blue.pkl",'rb') as f:
        blueRuler = pickle.load(f)
    with open(path+"rules_red.pkl",'rb') as f:
        redRuler = pickle.load(f)
    with open('column_transformer.pkl','rb') as f:
        c_tr = pickle.load(f)
    
    x = data.drop("moy")
    x["firstBlood"] = x.firstBlood.apply(lambda y: 1 if y=="blue" else 0)
    x = x[old_order].rename({"firstBlood":"blueFirstBlood"},axis=1)
    x = c_tr.transform(x.iloc[[i-1]])
    # print(len(blueRuler.rules_))
    # print(len(redRuler.rules_))
    def try_red(x,i):
        finished = False
        for j in range(len(redRuler.rules_)):
            if redRuler.predict_top_rules(x,1+j)==[1]:
                finished = True
                break
            # print(redRuler.predict_top_rules(x,j+i))
        if not finished:
            # print(redRuler.predict_top_rules(x,j+i))
            return []
        return redRuler.rules_[j:j+1]
    
    def try_blue(x,i):
        finished = False
        for j in range(len(blueRuler.rules_)):
            if blueRuler.predict_top_rules(x,j+i)==[1]:
                finished = True
                break
            # print(blueRuler.predict_top_rules(x,j+i))
        if not finished:
            # print(blueRuler.predict_top_rules(x,j+i))
            return []
        return blueRuler.rules_[j:j+1]
        
    if pred[i-1]>.5:
        team = "red"
        rl = try_red(x,i)
        # if rl == []:
        #     rl = 
    else:
        team = 'blue'
        rl = try_blue(x,i)
    # print(rl)
    txt = '\n'.join([translate_rule(x, team, w) for x in rl])
    if fake:
        # Si blueTotalGold est supérieur à 15929 et redTotalExperience est inférieur à 18426 alors l'équipe bleue gagnera.
        txt = "As blueTotalGold is higher than 15929 and redTotalExperienceis lower than 18426, blue team should win."
    if txt == '':
        txt = 'Pas de règle pour ce cas.'
    # col.write(txt)
    col.markdown("#### "+txt)

    
@st.cache
def make_df_pres():
    # print("make df pres is being used")
    l_ter = ["1e équipe à avoir infligé un kill.",
     'nombre de dragons (grand monstre) détruits par l’équipe.',
     'nombre de héraults (grand monstre) détruits par l’équipe.',
     'nombre de tourelles détruites par l’équipe.',
     'nombre de balises de vision placées par l’équipe.',
     'nombre de balises ennemies détruites par l’équipe.',
     'nombre de champions adverses tués par l’équipe.',
     'nombre d’assists (coopérations de champions à un kill) au sein de l’équipe.',
     'quantité totale de pièces d’or des champions.',
     'quantité totale d’expérience des champions.',
     'nombre de sbires ennemis détruits par les champions.',
     'nombre de petits monstres de la jungle détruits par les champions.'
     ]
    keys = index # old_order
    dic = {"features_names":keys,"description":l_ter}
    df_pres = pd.DataFrame(dic)
    df_pres = df_pres.set_index("features_names")
    # df_pres = df_pres.T[new_order].T #cassage de crâne
    return df_pres

