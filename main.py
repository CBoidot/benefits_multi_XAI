#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:06:56 2022

Ce "main" correspond à mon protocole v2, et fait suite aux "travaux
intermédiaire" (main_phase2.py) qui ont été menés pour évaluer la pertinence
de l'utilisation de ces données de LoL.
Principalement conçu à partir de codes pré-existants.

Le protocole d'expérience décrit ici : 
[to do]
Le but de l'expérience est d'évaluer l'utilité d'explications produites par la
librairie shap, dans le cadre d'une activité de pronostic sur des matchs d'e-
sport.

À utiliser avec l'environnement expenv1, stockée dans le même dossier.

La succession de l'expérience présentent une vingtaine de page.
Le déplacement entre les pages n'est pas toujours linéaire, et tous les
cas particuliers sont traités dans la fonction next.

Les données sont enregistrées dès que possibles : les données de décision
après les phases d'entraînement / évaluation, et les données de formulaire
à la fin de l'expérience.

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

pd.set_option("display.max_columns",40)
pd.set_option("display.max_rows",30)

st.set_page_config(layout="wide")

# USER = 0

#### initialisation des paramètres de session

path = "data_cache/" #"../Exp1-Streamlit/"

N_tr = 30 # total disponible
N_ev = 20 # pour une éval, donc *2 pour le total
# Le premier entraînement bloquera dès que l'utilisateur atteint N_ev+5.
# Pour le second, on décale l'ordre des données afin de ne pas remontrer (tout
# de suite) les mêmes données...
XP_MODE = "shap"


if "user" not in st.session_state:
    st.session_state["user"] = int(input("N° de l'expérience : "))
USER = st.session_state["user"]

if "cur_page" not in st.session_state:
    print(USER)
    st.session_state["cur_page"] = 1

if "cur_match" not in st.session_state:
    st.session_state["cur_match"] = 0

if "sv_first" not in st.session_state:
    if int(st.session_state.user)%2 == 0:
        st.session_state["sv_first"] = False 
    else:
        st.session_state["sv_first"] = True

if "completed" not in st.session_state:
    st.session_state["completed"] = [False]*5
    
if "part" not in st.session_state:
    st.session_state['part'] = 1
    st.session_state['train'] = True
    
st.session_state["with_sv"] = st.session_state.sv_first == st.session_state.part%2

i = st.session_state.cur_match
# if i in [10,11,12,13,14,15,16,17,18,19,25,26,27,28,29,35,36,37,38,39]:
#     st.session_state["with_sv"] = True
# else:
#     st.session_state["with_sv"] = False
# if st.session_state.sv_first and i<20:
#     st.session_state["with_sv"] = not st.session_state["with_sv"]

if "reponses_num" not in st.session_state:
    st.session_state["reponses_num"] = pd.DataFrame(np.zeros((0,3)),
                            columns=["rep","time_print","time_click"])
    
if "click_list" not in st.session_state:
    st.session_state["click_list"] = []
    
if "reponses_form" not in st.session_state:
    st.session_state["reponses_form"] = pd.DataFrame(
                                            pd.Series(np.zeros((0))),
                                            columns=["rep"])

if "t_print" not in st.session_state:
    st.session_state["t_print"] = None
# utilisé pour obtenir l'instant où la page est affichée.

                     
r1, r2, r3, r4, r5 = None, None, None, None, None    

clickable = False


## Format réponses

options_ope = ["l’équipe bleue va certainement gagner",
            "l’équipe bleue va vraisemblablement gagner",
            "l’équipe bleue a un léger avantage sur l’équipe rouge",
            "je ne sais pas",
            "l’équipe rouge a un léger avantage sur l’équipe bleue",
            "l’équipe rouge va vraisemblablement gagner",
            "l’équipe rouge va certainement gagner"]

interp_ope = {"l’équipe bleue va certainement gagner":0,
            "l’équipe bleue va vraisemblablement gagner":0,
            "l’équipe bleue a un léger avantage sur l’équipe rouge":0,
            "l’équipe rouge a un léger avantage sur l’équipe bleue":1,
            "l’équipe rouge va vraisemblablement gagner":1,
            "l’équipe rouge va certainement gagner":1}

temp_jeu = ["Il se passe rarement plus de quelques jours sans que je ne joue.",
                "Je joue au moins une fois par mois dernièrement.",
                "Il arrive facilement qu'un mois passe sans que je ne joue.",
                "J'ai fait une pause d'un an ou plus dernièrement."]

temp_visio = ["Il se passe rarement plus de quelques jours sans que je ne regarde un match.",
                "J'en regarde au moins une fois par mois dernièrement.",
                "Il arrive facilement qu'un mois passe sans que je ne ne regarde de match.",
                "J'ai fait une pause d'un an ou plus dernièrement."]


rangs = ["Non classé", "Fer", "Bronze", "Argent", "Or", "Platine", "Diamant",
         "Maître", "Grand Maître", "Challenger"]

datascience = ["aucune", "des bases en statistiques", "une formation math-info",
               "je travaille dans la data science",
               "expert en ML, avec des connaissances en XAI"]

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

q1 = ["L’interface d'analyse comprenait toutes les informations pertinentes \
      pour m’aider à prendre une décision.",
      "L’interface d'analyse m’a permis de prendre une décision plus rapide.",
      "L’interface d'analyse m’a été utile pour faire un pronostic.",
      "L’interface était facile à utiliser.", "Comment s’est passée votre \
          expérience avec l’interface d'analyse du modèle ? Comment l’avez-vous utilisée ?",
      "L’interface d'analyse m’a permis de saisir comment l’IA fonctionnait.",
      "L’IA utilisée est capable de faire de proposer de bons pronostics.",
      "L’interface d’analyse expliquait bien le modèle, de façon claire et concise.",
      "Qu’attendriez-vous d’une IA qui tente de vous expliquer sa décision ?",
      "Si vous faisiez réellement des pronostics à 10 minutes, aimeriez-vous \
          avoir un modèle pour vous assister ?", "Si vous faisiez réellement \
       des pronostics à 10 minutes, aimeriez-vous avoir accès aux données moyennes ?",
      "Si vous faisiez réellement des pronostics à 10 minutes, aimeriez-vous \
          avoir une interface d'analyse de modèle ?",
      "Avez confiance en le développement futur de l'IA ?", "Seriez-vous prêt \
          à utiliser un système d’IA similaire, avec xp° des résultats, dans un autre contexte ?",
      "Quelles sont vos attentes vis-à-vis de l’utilisation de l’IA dans un cadre d’application similaire ?",
      "Quel est votre niveau de formation en informatique / sciences de l’ingénieur ?",
      "Avez-vous des connaissances en Intelligence Artificielle ?",
      "Avez-vous d’autres connaissances liées à l’AI ou l’XAI en particulier ?"]
q2 = ["À quelle fréquence jouez-vous à League of Legends ?",
     "Quels rôles avez-vous déjà joué, jouez-vous usuellement ?",
     "Avez-vous déjà joué en équipe avec des amis ? (et si oui, combien)",
     "Depuis combien d'années jouez-vous à des MOBA ?",
     "Quel est votre rang actuel dans LoL ?",
     "À quelle fréquence regardez-vous des matchs compétitifs ?",
     "Depuis combien d'année vous êtes-vous intéressé à la méta de LoL ? (0 si vous l'ignorez)",
     "Avez-vous déjà publié du contenu grâce à LoL (vidéo, article) ? \
         Gagnez-vous de l'argent graĉe à LoL ?",
     "Pensez-vous être capable de bien estimer les probabilités de victoire des équipes ?",
     "Vous êtes : ", "Quel âge avez-vous : ", "Y a-t-il d'autres points à \
         relever sur votre profil ou sur le déroulé de l'expérience ?"]


### Chargement des données ###

@st.cache
def init_data_tr():
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
    
    labelshuffled = labels.sample(frac=1,random_state=USER)
    # phase de tri : chaque bloc de 5 doit avoir au moins 1 rouge et 1 bleu
    for i in range(N_tr):
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
    labels = labels.loc[final_index]
    m = df_select.loc["moy"]
    df_select = df_select.loc[final_index]
    df_select.loc["moy"] = m
    
    with open("results/shuffle"+str(USER)+".pkl",'wb') as f: #path+
        pickle.dump(labels,f)
    
    labels = labels.reset_index(drop=True)
    # sans quoi, je ne peux les comparer pour donner une note à l'utilisateur
    return df_select, labels
df_tr, labels = init_data_tr()

@st.cache
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
#     with open(path+"results/redondance_"+str(USER)+".pkl",'wb') as f:
#         pickle.dump(to_insert,f)
#     df_bis = df_select.iloc[:N].reset_index(drop=True)
#     for i in range(5):
#         df_bis.loc[N+2*i] = df_select.iloc[10+i] 
#         df_bis.loc[N+2*i+1] = to_insert.iloc[i]
#     df_bis = pd.concat([df_bis,df_select.reset_index(drop=True).iloc[N+5:50]]) \
#         .reset_index(drop=True)
#     df_bis.loc['moy'] = df_select.loc['moy']
    return df_select
df_ev = init_data_ev()

@st.cache
def init_sv():
    '''
    xpo_tr : explication telle que définies dans shap. xpo.data est en old order
    xpo_ev : explication telle que définies dans shap. xpo.data est en old order
    pred_tr : np.array
    pred_ev : np.array
    '''
    with open(path+"shap_tr.pkl",'rb') as pf:
        xpo_tr = pickle.load(pf)
    with open(path+"rf_tr.pkl",'rb') as pf:
        pred_tr = pickle.load(pf).values
    with open(path+"shap_ev.pkl",'rb') as pf:
        xpo_ev = pickle.load(pf)
    with open(path+"rf_ev.pkl",'rb') as pf:
        pred_ev = pickle.load(pf).values
    # correction
    xpo_tr.data = xpo_tr.data[old_order]
    xpo_ev.data = xpo_ev.data[old_order]
    
    # order !!
    # un moyen d'avoir la permutation
    perm = [xpo_tr.data.index.get_indexer_for([df_tr.index[i]])[0] for i in range(N_tr)]
    xpo_tr = xpo_tr[perm]
    # xpo_tr[perm].data.index==df_tr.index.drop("moy") # True!
    pred_tr = pred_tr[perm]
    return xpo_tr, xpo_ev, pred_tr, pred_ev
xpo_tr, xpo_ev, pred_tr, pred_ev = init_sv()



###################### N.E.X.T ######################


def add_form(n):
    """
    n: int, nombre de lignes de formulaire à entrer
    """
    k = st.session_state.reponses_form.shape[0]
    if n==1: #
        st.session_state['reponses_form'].loc[k] = r1
    elif n==3: #
        st.session_state['reponses_form'].loc[k] = r1
        st.session_state['reponses_form'].loc[k+1] = r2
        st.session_state['reponses_form'].loc[k+2] = r3
    elif n==5: #
        st.session_state['reponses_form'].loc[k] = r1
        st.session_state['reponses_form'].loc[k+1] = r2
        st.session_state['reponses_form'].loc[k+2] = r3
        st.session_state['reponses_form'].loc[k+3] = r4
        st.session_state['reponses_form'].loc[k+4] = r5
    elif n>5:
        rep = pd.DataFrame(np.array(r1).T,columns=["rep"])
        st.session_state['reponses_form'] = \
                pd.concat([st.session_state['reponses_form'],rep])
    # print(st.session_state['reponses_form'])

def nextpage(back=0):
    global clickable 
    if not clickable:
        time.sleep(.8) # print("Ne cliquez pas si vite !")
    else:
        clickable = False
        sscp = st.session_state.cur_page
        if sscp==6: # début des cas spéciaux, sinon +1
            line = [r1, st.session_state.t_print, time.time()]
            st.session_state['reponses_num'].loc[st.session_state.cur_match] = line
            st.session_state['t_print'] = None
            st.session_state.cur_match += 1
            c2 = st.session_state.cur_match%5==0
            if c2:
                st.session_state.cur_page += 1
        elif sscp == 7:
            if back==1:
                st.session_state.cur_page = 6 #
            else:
                st.session_state.cur_page += 1
        elif sscp==8:
            add_form(3)
            # print("part : {}".format(st.session_state.part))
            if st.session_state.train:
                st.session_state.train = False
                # print("on passe en eval")
                if st.session_state.part==1:
                # "end_tr1" not in st.session_state:
                    st.session_state["end_tr1"] = st.session_state.cur_match
                st.session_state.cur_match = 0
                st.session_state.cur_page += 1
                if st.session_state.part==2:
                    st.session_state['cur_match'] = N_ev
            else:
                st.session_state.train = True
                # print("on passe en train")
                st.session_state.cur_page = 12
                st.session_state.cur_match = 0
                st.session_state['isok'] = False
        # elif sscp==9: # pas un cas spécial
        #     st.session_state.reponses_num = df.drop(df.index)
        #     st.session_state.cur_match = 0
        #     st.session_state.cur_page = 6
        elif sscp==10:
            line = [r1, st.session_state.t_print, time.time()]
            st.session_state['reponses_num'].loc[st.session_state.cur_match] = line
            st.session_state['t_print'] = None
            st.session_state.cur_match += 1
            st.session_state.cur_page += 1
        elif sscp==11:
            if st.session_state.cur_match%N_ev==0:
                st.session_state.cur_page = 8
            else:
                st.session_state.cur_page -= 1
        elif sscp==12:
            if st.session_state.part==1:
                st.session_state.cur_page = 3
                st.session_state.cur_match = 0
                st.session_state.part = 2
                # print("Part {}".format(st.session_state.part))
            else:
                st.session_state.cur_page += 1
                
        else: 
            if sscp >= 12: #pages finales de formulaire
                if sscp==17:
                    add_form(8)
                elif sscp<16:
                    add_form(5)
                else:
                    add_form(3)
            st.session_state.cur_page += 1
        st.session_state.click_list.append(time.time())        
        st.session_state['t_print'] = None
        st.session_state["completed"] = [False]*5
        time.sleep(.25)
        # print('train' if st.session_state.train else 'eval')

def cheatcode(i):
    '''
    Conçu pour debugguer plus vite
    '''
    st.session_state.cur_page = i
    st.session_state["completed"] = [False]*5


### factorisation : fonctions d'affichage ###


index = ['firstBlood', 'Dragons', 'Heralds',
       'TowersDestroyed', 'WardsPlaced', 'WardsDestroyed',
       'Kills', 'Assists', 'TotalGold', 'TotalExperience',
       'TotalMinionsKilled','TotalJungleMinionsKilled']
np.set_printoptions(suppress=True)

def display_match(i,df=None,col=st):
    """attention, les i que vous entrez commencent à 1"""
    # print("number: {}".format(i))
    if df is None:
        if st.session_state.train:
            df = df_tr
        else:
            df = df_ev
        # df.iloc[:5]
    if i==-1: # 21 mars 2023
        i = -2
    if False: # oneliner version, utilisée précédemment
        tab = df.iloc[[i,-1]].T
        tab.columns = ["match","moy"]
        col.table(tab.astype(str))
    else:
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
            new_df.astype(str) # print(...)
        else:
            col.table(new_df.astype(str))

def display_score(i,pred=None,col=st,wording=1):
    """TO DO: intégrer pred""" # /!\ # ???
    if pred is None:
        if st.session_state.train:
            pred = pred_tr
        else:
            pred = pred_ev
    if wording == 1:
        txt = "### Victoire prédite : équipe _{}_\n #### Probabilité estimée : _{}_%"
    elif wording == 2:
        txt = "### Pronostic IA : équipe _{}_\n #### Probabilité estimée : _{}_%"
    if pred[i-1]>.5:
        team, percentage = "rouge", str(int(pred[i-1]*100))
    else:
        team, percentage = 'bleue', str(int((1-pred[i-1])*100))
    col.markdown(txt.format(team,percentage))

def display_shap(i,xpo=None,col=st):
    
    if xpo is None:
        if st.session_state.train:
            xpo = xpo_tr
        else:
            xpo = xpo_ev
    shap.plots.bar(xpo[i-1],show_data=True,show=False)#st.session_state.cur_match
    fig = plt.gcf()
    col.pyplot(fig)

@st.cache
def make_df_pres():
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
df_pres = make_df_pres()



##############################################################################
#               ICI COMMENCE L'INTERACTION AVEC L'UTILISATEUR                #
##############################################################################



# liste des pages

if st.session_state.cur_page == 0:
    st.title("Perdu !")

if st.session_state.cur_page == 1:
    
    ### Présentation du problème
    st.title("Présentation de votre mission")
    "***À lire impérativement par tous.***\n"
    
    "Bonjour, et merci de participer à notre expérience."
    "On tente de prédire l'équipe gagnante lors de matchs de League of Legends (LoL). \
        L'objectif est de parier à partir de statistiques de jeu prises 10 minutes après le départ. \
            Il s’agit de matchs classés solo de rang Diamant I à Master, durant la saison 10 de LoL \
                (les joueurs ne se connaissent pas avant le début du match : les équipes sont formées de façon impartiales)."
    #    "Le programme en cours est une expérience en 4 phases afin de mesurer \
    "L'expérience va se dérouler en 4 phases, et vise à mesurer \
        vos performances prédictives sur nos données de match dans différentes \
            conditions : seul ou assisté par une IA. Vous aurez 2 phases d'entraînements \
                pour pouvoir vous approprier la tâche, et vous la réappropier \
                    quand le contexte IA change."
    "La précision des réponses est aussi importante que la vitesse de réponse."
    # /!\ définir plus d'enjeux ? /!\
    "Cliquez sur START pour continuer."
    
    st.button("START>>>",key='nekst',on_click=nextpage) 
    # i = st.slider("aller à la page")  ## pour débugger
    # st.button("GO!",on_click=cheatcode,args=[i])  ## débugger
    
    # display_match(30,df_tr) # le 15 nov22, pour debug

elif st.session_state.cur_page == 2:
    
    st.title("Présentation des données statistiques")
    
    left_column, right_column = st.columns(2)
    with left_column:
        "Durant l'expérience, vous aller devoir traiter des données au format \
            ci-contre. Vous y trouverez les relevés de l'équipe bleue, de l'équipe \
                rouge."
        "La colonne 'mean' contient la moyenne statistique de la valeur \
            mesurée, à partir de l'ensemble des matchs à disposition."
        "À gauche, vous avez une description des colonnes que vous pouvez \
            masquer si besoin."
    display_match(1,col=right_column)
        
    st.sidebar.table(df_pres)
    # "Le programme que vous venez de lancer a pour but de fournir un entraînement à cette tâche."
    # "Vous aurez plusieurs séries de match à évaluer, sous la forme de relevés de données de jeu. \
    #     Votre évaluation revient à une décision (une sorte de pari) que vous seriez prêt à prendre vis-à-vis du match."
    # # "Le format des données est présenté à la page suivante."
    # "Pour vous préparer, le programme vous propose d'abord d'observer des données de matchs, sans évaluation."
    st.session_state["train"] = True
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage) 
    

elif st.session_state.cur_page == 3:
    
    st.title("Instructions pour l'entraînement")
    left_column, right_column = st.columns(2)
    with left_column:
        if st.session_state.part==1:
            if st.session_state.with_sv: 
                "Vous allez commencer en étant assisté par une IA."
                "Celle-ci produit un score de prédiction continu, sous forme de \
                    probabilité de victoire. Vous aurez accès à cette prédiction \
                        ainsi qu'à une interface d'analyse."
                "Cette interface montre les valeurs les plus importantes dans la \
                    décision du modèle, et dans quel sens chacune a contribué."
                "Ces contribution sont représenté ci-dessous par des barres \
                    de longueur proportionnelle à l'importance."
                "La barre est rouge si la variable a contribué au bon score de \
                    l'équipe rouge (et respectivement bleue pour l'équipe bleue)."
                display_shap(1,col=right_column)
            else:
                "Vous allez commencer en utilisant uniquement les données fournies. \
                    Les recommandations de l'IA viendront dans une autre phase."
    
        elif st.session_state.part==2:
            if st.session_state.with_sv: 
                "Vous allez maintenant être assisté par une IA."
                "Celle-ci produit un score de prédiction continu, sous forme de \
                    probabilité de victoire. Vous aurez accès à cette prédiction \
                        ainsi qu'à une interface d'analyse"
                "Cette interface montre les valeurs les plus importantes dans la \
                    décision du modèle, et dans quel sens chacune a contribué"
                "Ces contribution sont représenté ci-dessous par des barres \
                    de longueur proportionnelle à l'importance."
                "La barre est rouge si la variable a contribué au bon score de \
                    l'équipe rouge (et respectivement bleue pour l'équipe bleue)."
                display_shap(1,col=right_column)
            else:
                "Vous allez maintenant devoir vous passer de l'IA."
        else:
            "bug: partie >2"
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)    


elif st.session_state.cur_page == 4:

    left_column, right_column = st.columns(2)
    if st.session_state.part==2:
        ajust = st.session_state.end_tr1 # hé, il faut vivre avec son temps.
    else:
        ajust = 0
        
    with right_column:
        
        i = 0 
        # "Pour prendre en main l'expérience, nous vous proposons de regarder les données de 5 matchs. \
        # Après quoi, vous aurez un entraînement rapide avant de commencer la tâche."
        "Regardez les données des 5 matchs avant de passer à la suite."
        i = st.selectbox("Choisissez le match à afficher",
                         [1,2,3,4,5]) # list(range(1,31)))#
        vus = st.session_state.completed
        vus[i-1] = True
        
        winner = '_rouge_' if labels.iloc[i-1+ajust] else '_bleue_'
        st.markdown("## Équipe gagnante : " + winner)
        if st.session_state.with_sv:
            display_score(i+ajust)
            display_shap(i+ajust)
        # st.write(st.session_state.completed) # TO DO : effacer ça
        if reduce(lambda x,y: x and y, vus):
            increment = st.button("NEXT>>>",on_click=nextpage)
    
    
    left_column.title("Match d'observation n°"+str(i))
    display_match(i+ajust-1,col=left_column)
    
    st.sidebar.table(df_pres)
    # st.button("TRICHE>>>",key='nekst',on_click=nextpage) ## debug


elif st.session_state.cur_page == 5:
    
    if st.session_state.part==1:
        st.title("Exprimer une prédiction")
        
        "À présent, vous allez devoir exprimer votre pronostic sur 5 matchs consécutivement. Suite à cela, vous pourrez voir vos potentielles erreurs."
        if st.session_state.train:
            "Vous allez pouvoir exprimer votre réponse sur une échelle d'incertitude/certitude en faveur d'une équipe ou de l'autre."
        "Suite à cette première évaluation, soit vous continuerez automatiquement l'entraînement, soit on vous proposera de passer à la tâche réelle."
                # "Vous pourrez exprimer votre pronostic sur une échelle à 7 crans, \
                # selon votre degré de certitude, avec les sliders suivants. \u2193"
                
        if st.session_state.train:
            r1 = st.select_slider("exemple de slider.",
                                  options_ope,"je ne sais pas")    
    else:
        st.title("Transition")
        
        "Vous pouvez passer à la suite de l'entraînement."
    
    st.session_state.cur_match = 5 # /!\ organisation à questionner /!\
    
    st.button("NEXT>>>",on_click=nextpage)
    
    
    ## Première page de prise de décision, pour l'entraînement
elif st.session_state.cur_page == 6: #or st.session_state.cur_page == 10:
    
    left_column, right_column = st.columns(2) # ou 3 ?
    if st.session_state.part==2:
        ajust = 5 + st.session_state.end_tr1 # hé, il faut vivre avec son temps.
    else:
        ajust = 5 # c'est dû à l'exposition avant
    
    with right_column:
        
        i = st.session_state.cur_match -4
        if st.session_state.with_sv:
            display_score((i+ajust)%N_tr,col=right_column)
            display_shap((i+ajust)%N_tr,col=right_column)
        title = 'Quel est votre pronostic ?'
        vus = st.session_state.completed
        vus[(i-1)%5] = True
        r1 = st.select_slider(title,options_ope,"je ne sais pas",key="slide")
        increment = st.button("NEXT>>>",on_click=nextpage)
    
    left_column.title("Match n°"+str(i))
    display_match((i+ajust-1)%N_tr,col=left_column) # /!\ cf page 4
    st.sidebar.table(df_pres)
    
    
    ## retour à l'utilisateur
elif st.session_state.cur_page == 7:
    
    left_column, right_column = st.columns(2) # ou 3 ?
    if st.session_state.part==2:
        ajust = 5 + st.session_state.end_tr1 # hé, il faut vivre avec son temps
    else:
        ajust = 5
    
    ## score computation
    i_ = st.session_state.cur_match -10
    extrait = st.session_state.reponses_num.loc[i_+5:i_+9,'rep']
    exprimes = extrait.apply(lambda x: False if x=="je ne sais pas" else True)
    numerateur = exprimes.sum()
    choix = extrait[exprimes].apply(lambda x: interp_ope[x])
    # # labels.loc[i+5:i+9][exprimes]
    dividende = (choix.values == \
                 labels.loc[(i_+ajust)%N_tr:(i_+ajust+4)%N_tr].values[exprimes]
                 ).sum()
    
    isok = dividende>=3 and numerateur-dividende<=1
    if "isok" not in st.session_state:
        st.session_state['isok'] = isok
    else:
        st.session_state['isok'] = isok or st.session_state.isok
    
    with right_column:
        
        st.title('Votre score : ' + str(dividende)+" / "+str(numerateur))
        # /!\ je réutilise i à une autre fin
        i = st.selectbox("Choisissez le match à afficher",
                         [i_+1,i_+2,i_+3,i_+4,i_+5])
        vus = st.session_state.completed
        vus[(i-1)%5] = True
        
        winner = 'rouge' if labels.iloc[(i+ajust-1)%N_tr] else 'bleue'
        st.title("Équipe gagnante : " + winner)
        predi = st.session_state.reponses_num.loc[i+4,'rep']
        st.title("Votre pronostic : " + predi)
        if st.session_state.with_sv:
            display_score((i+ajust)%N_tr,col=right_column,wording=2)
            display_shap((i+ajust)%N_tr,col=right_column)
        
        if st.session_state.cur_match >= N_ev+4:
            "Plus de match disponibles."
            st.button("Suite>>>",on_click=nextpage)
        else:
            if not isok:
                "Vous pouvez mieux faire."
                "Regardez vos erreurs et recommencez."
                if reduce(lambda x,y: x and y, vus):
                    st.button("Recommencer>>>",on_click=nextpage,args=[1])
                    if st.session_state.isok:
                        "(Puisque vous avez déjà réussi l'entraînement \
                            vous pouvez continuer si vous le souhaitez.)"
                        st.button("Suite>>>",on_click=nextpage)
            else:
                "Bien joué ! Vous pouvez passer à la suite si vous voulez, \
                    sinon vous pouvez refaire une série."
                st.button("Recommencer>>>",on_click=nextpage,args=[1])#,key="ye")
                if st.session_state.isok:
                    st.button("Suite>>>",on_click=nextpage)
    
    display_match((i+ajust-1)%N_tr,col=left_column)
    if not isok:
        left_column.write("Vous devez réussir au moins 75% de vos prédictions, \
                          avec moins de trois 'je ne sais pas'.")
    st.sidebar.table(df_pres)
    # display_match(0)
    
    
elif st.session_state.cur_page == 8:
    
    if st.session_state.train:
        st.title("Fin de la phase d'évaluation")
    else:
        st.title("Fin de la phase d'entraînement")
    
    "Nous allons recueillir vos ressentis."
    
    r1 = st.select_slider("Pensez-vous avoir progressé au cours de l’utilisation ?",
                          nondutout,"indécis")
    
    r2 = st.select_slider("Étiez-vous très concentré durant l’expérience ?",
                          nondutout,"indécis")
    
    r3 = st.select_slider("L’expérience vous a-t-elle fatigué ?",
                          nondutout,"indécis")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)


elif st.session_state.cur_page == 9:
    
    st.title("La phase d'évaluation va commencer")
    
    df = st.session_state.reponses_num
    df.to_csv("results/reponses_matchs_tr_p" + str(st.session_state.part)
              + '_' +  str(USER) + ".csv", index=False)
    st.session_state['reponses_num'] = pd.DataFrame(np.zeros((0,3)),
                            columns=["rep","time_print","time_click"])
    
    "Vous allez passer à l'évaluation réelle de votre capacité de pronostic.\n"
    "Objectif : prédire l'équipe gagnante d'un match de League of Legends \
        à partir de données statistiques à 10 minutes du match."
    st.write("Vous aurez {} matchs à évaluer : **vous pouvez prendre une pause entre \
        deux pronostics** au besoin, mais une fois que l'écran de match s'affiche, \
            tentez de **répondre au plus vite**.".format(N_ev))
    # "Évitez autant que possible de passer plus d'une minute sur un match."
    
    st.button("START>>>",key='nekst',on_click=nextpage)
    
    
    ## Seconde page de prise de décision, pour l'évaluation
elif st.session_state.cur_page == 10:
    
    # je me permets ce copié-collé à cause de l'indexx décalé de 5
    left_column, right_column = st.columns(2)
    with right_column:
        
        i = st.session_state.cur_match
        # st.title("Match n°"+str(i+1))
        if st.session_state.with_sv:
            display_score(i,col=right_column)
            display_shap(i,col=right_column)
        title = 'Quel est votre pronostic ?'
        r1 = st.select_slider(title,options_ope,"je ne sais pas",key="slide")
        increment = st.button("NEXT>>>",on_click=nextpage)
    
    left_column.title("Match n°"+str(i+1))
    display_match(i-1,col=left_column)
    
    st.sidebar.table(df_pres)
    
    
elif st.session_state.cur_page == 11:
    
    left_column, right_column = st.columns(2)
    if st.session_state.cur_match%N_ev!=0:
        left_column.title("Paré ?")
        right_column.button("NEXT>>>",on_click=nextpage) #,args=[6])
    else:
        left_column.title("Fin des matchs !")
        right_column.button("NEXT>>>",on_click=nextpage)
    
    st.sidebar.table(df_pres)
    

elif st.session_state.cur_page == 12:
    
    st.title("Transition")
    
    # print("part = {}".format(st.session_state.part))
    df = st.session_state.reponses_num
    df.to_csv("results/reponses_matchs_ev_p" + str(st.session_state.part)
              + '_' +  str(USER) + ".csv", index=False)
    if st.session_state.part%2:
        "Vous allez pouvoir passer à la deuxième partie."
        st.session_state['reponses_num'] = pd.DataFrame(np.zeros((0,3)),
                            columns=["rep","time_print","time_click"])
        
    else:
        "Il reste une dernière étape : remplir un formulaire rapide (moins de 5 minutes)."
    st.button("NEXT>>>",on_click=nextpage)
        
    
############################## LES FORMULAIRES ##############################
    
elif st.session_state.cur_page == 13:
    
    st.title("Questions sur l'interface d'analyse du modèle informatique")
    
    "Sur la moitié des matchs, vous avez eu accès à un graphique transcrivant \
        la recommandation du modèle."
    "Êtes-vous d'accord avec les propositions suivantes :"
    
    r1 = st.select_slider(q1[0],
                          pasdaccord,"neutre")
    r2 = st.select_slider(q1[1],
                          pasdaccord,"neutre")
    r3 = st.select_slider(q1[2],
                          pasdaccord,"neutre")
    r4 = st.select_slider(q1[3],
                          pasdaccord,"neutre")
    r5 = st.text_input(q1[4])
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    
    
elif st.session_state.cur_page == 14:
    
    st.title("Questions sur l'algorithme")
    "L'algorithme d'aide au pronostic utiliserait correspond à ce que l'on \
        appelle communément un modèle d'intelligence artificielle (IA)."
    "Exprimez votre accord avec les propositions suivantes :"
    
    r1 = st.select_slider(q1[5],
                           pasdaccord,"neutre")
    
    r2 = st.select_slider(q1[6],
         pasdaccord,"neutre")
    
    r3 = st.select_slider(q1[7],
                          pasdaccord,"neutre")
    
    r4 = st.text_input(q1[8])
    
    r5 = st.select_slider(q1[9],
                          nondutout, 'indécis')
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    

elif st.session_state.cur_page == 15:
    
    st.title("Utilisation de l'IA et d'interfaces explicatives")
    
    r1 = st.select_slider(q1[10],
                          nondutout, 'indécis')
    
    r2 = st.select_slider(q1[11],
                          nondutout, "indécis")
    
    r3 = st.select_slider(q1[12],
                          nondutout,'indécis')
    
    r4 = st.select_slider(q1[13],
                          nondutout,'indécis')
    
    r5 = st.text_input(q1[14])
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    

elif st.session_state.cur_page == 16:
    
    st.title("Expertise en IA")
    
    r1 = st.radio(q1[15], ["Aucun", "Bac spé math/info", "Bac+2", "Bac+3 (équivalent licence)", 
                           "Bac+4", "Bac+5 (équivalent master)", "Bac+6", "supérieur"])
    
    r2 = st.radio(q1[16], ["Aucune", "en amateur (par la vulgarisation)", "passionné d’informatique et de statistique (connaissances techniques)", 
                    "jeune pro (M2 ou doctorant)", "confirmé (plus de trois ans de travail dans le domaine du Machine Learning"])
    
    r3 = st.text_input(q1[17])
                       
    # r4 = st.select_slider("Distinguez-vous intuitivement les trois fonctions suivantes : la probabilité des données, la probabilité de victoire de l’équipe rouge, et la fonction calculée par le modèle de prédiction ?",
    #                       nondutout, "indécis")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)


elif st.session_state.cur_page == 17:
    
    st.title("Questions sur votre expérience avec League of Legends")
    
    r1 = [0]*9
    
    l ="À quelle fréquence jouez-vous à League of Legends ?"
    r1[0] = st.selectbox(l,temp_jeu)
    
    l="Quels rôles avez-vous déjà joué, jouez-vous usuellement ?"
    r1[1] = st.text_input(l)
    
    l="Avez-vous déjà joué en équipe avec des amis ? (et si oui, combien)"
    r1[2] = st.selectbox(l,["juste en solo",1,2,4])
    
    l="Depuis combien d'années jouez-vous à des MOBA ?"
    r1[3] = st.number_input(l, max_value=21,format="%d")
    
    l="Quel est votre rang actuel dans LoL ?"
    r1[4] = st.selectbox(l, rangs)
    
    l="À quelle fréquence regardez-vous des matchs compétitifs ?"
    r1[5] = st.selectbox(l,temp_visio)
    
    l="Depuis combien d'année vous êtes-vous intéressé à la méta de LoL ? (0 si vous l'ignorez)"
    r1[6] = st.number_input(l, max_value=14,format="%d")
    
    l="Avez-vous déjà publié du contenu grâce à LoL (vidéo, article) ? Gagnez-vous de l'argent graĉe à LoL ?"
    r1[7] = st.text_input(l)
    
    # l="Avez-vous des connaissances en science des données ? (choisissez au plus proche)"
    # r1[8] = st.select_slider(l,datascience)
    
    r1[8] = st.select_slider("Pensez-vous être capable de bien estimer les probabilités de victoire des équipes ?",
                          nondutout,"indécis")
    #"Pensez-vous être capable d’estimer la probabilité de victoire de l’équipe rouge ?"
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    
    
elif st.session_state.cur_page == 18:
    
    st.title("Informations générales")
    
    r1 = st.radio("Vous êtes : ", ["un homme","une femme","préfère ne pas répondre"])
    
    r2 = st.text_input("Quel âge avez-vous : ")
    
    r3 = st.text_input("Y a-t-il d'autres points à relever sur votre profil ou sur le déroulé de l'expérience ?")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    
    
elif st.session_state.cur_page == 19:
    
    st.title("Félicitation, vous avez fini !")
    
    "\nPour de vrai, cette fois. ;)"
    
    with open("results/times_tr_" + str(USER) + ".pkl",'wb') as f:
        pickle.dump(st.session_state.click_list,f)
    
    # with open("results/times_" + str(st.session_state.user) + ".pkl",'wb') as f:
    #     pickle.dump(st.session_state.click_list,f)
    
    # on reindexe par les questions de formulaire.
    st.session_state['reponses_form'] = \
        st.session_state.reponses_form.reset_index(drop=True).rename(
            index=lambda x:  (list(range(12))+q1+q2)[x])
    
    st.session_state.reponses_form.to_csv(
        "results/reponses_formulaire_" + str(st.session_state.user) + ".csv")

    
elif st.session_state.cur_page == 999:
    
    "Perfectly balanced..."
    "As everything should be."
    


if st.session_state.t_print is None: # permet le calcul du temps d'affichage.
    st.session_state['t_print'] = time.time()
st.write("page : "+str(st.session_state.cur_page))
clickable = True

