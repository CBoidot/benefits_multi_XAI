#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:06:56 2022

Copie du main.py au 21 nov 2022, ce fichier a pour but de développer et 
montrer mes alternatives explicatives.

La stratégie sera de se placer en "page 0" pour faire  ce qu'on veut tout en 
gardant la possibilité de passer à la suite de l'expé.

/!\ Ma première action: commenter les .dump et les to_csv

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

from utils.d4data import (init_data_ev, init_data_tr, init_likelihood, 
                          init_simil, init_fi)

from utils.f4factorized import (display_match, display_score, display_shap,
                            display_likelihood, display_arg, 
                            feature_selection_shap, display_pastilles,
                            display_nearest,display_rule,make_df_pres)



pd.set_option("display.max_columns",40)
pd.set_option("display.max_rows",30)

st.set_page_config(layout="wide")

# USER = 0

#### initialisation des paramètres de session

path = "" #"../Exp1-Streamlit/"

N_tr = 30 # total disponible
N_ev = 20 # pour une éval, donc *2 pour le total
# Le premier entraînement bloquera dès que l'utilisateur atteint N_ev+5.
# Pour le second, on décale l'ordre des données afin de ne pas remontrer (tout
# de suite) les mêmes données...
# XP_MODE = "shap"

# /!\ TO DO
USER = 77

#%% non
# if "user" not in st.session_state:
#     st.session_state["user"] = int(input("N° de l'expérience : "))
st.session_state["user"] = 77
# # USER = st.session_state["user"]
#%% oui

if "cur_page" not in st.session_state:
    print(USER)
    st.session_state["cur_page"] = 0

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
    
st.session_state["with_sv"] = \
    st.session_state.sv_first == st.session_state.part%2

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

if "aide" not in st.session_state:
    st.session_state["aide"] = False

#%% init variables
                     
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

df_tr, labels = init_data_tr(USER,N_tr)
df_ev = init_data_ev()
xpo_tr, xpo_ev, pred_tr, pred_ev = init_fi(N_tr,df_tr)
likeli_tr,likeli_ev = init_likelihood()
near_tr,near_ev = init_simil()
near1_tr,near1_ev, ann1_tr, ann1_ev = init_simil(1,alter=True)
near1_tr = near1_tr.reindex(index=df_tr.drop('moy').index)
ann1_tr = ann1_tr.reindex(index=df_tr.drop('moy').index)
lime_tr, lime_ev, _, _ = init_fi(N_tr,df_tr,"lime")


###################### N.E.X.T ######################
#%% pagination 

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
        st.session_state["aide"] = False
        time.sleep(.25)
        # print('train' if st.session_state.train else 'eval')

def cheatcode(i):
    '''
    Conçu pour debugguer plus vite
    '''
    st.session_state.cur_page = i
    st.session_state["completed"] = [False]*5


# factorisation : fonctions d'affichage #=> cf utils.f4factorized



np.set_printoptions(suppress=True)


df_pres = make_df_pres()




if st.session_state.train:
    df = df_tr
    pred = pred_tr
    sv = xpo_tr
    like = likeli_tr
    near = near_tr
    near1 = near1_tr
    lime = lime_tr
else:
    df = df_ev
    pred = pred_ev
    sv = xpo_ev
    like = likeli_ev
    near = near_ev
    near1 = near1_ev
    lime = lime_ev

##############################################################################
#%%             ICI COMMENCE L'INTERACTION AVEC L'UTILISATEUR              %%#
##############################################################################


# liste des pages

if st.session_state.cur_page == 0:
    
    # Bienvenue Néo.
    # Tu vas découvrir mes stratégies expérimentales de développement
    # d'alternatives explicatives au sacro-saint SHAP.
    
    left_column, right_column = st.columns(2) # ou 3 ?
    
    mode = left_column.selectbox("Mode de l'interface", ["neutre","bar-SHAP",
                                 "texte","probas","pastilles","simil2",
                                 "rule","simil1",'lime-line','lime-flot',
                                 'lime-bar','lime-pastille','fake_rule'])

    with right_column:
        
        ajust = 0
        i = st.selectbox("Choisissez le match à afficher",
                         list(range(1,N_tr+1)))
        
        
        # predi = st.session_state.reponses_num.loc[i+4,'rep']
        # st.title("Votre pronostic : " + predi)
        if mode != "neutre":
            if mode == "bar-SHAP":
                display_shap((i+ajust)%N_tr,sv,col=right_column)
            elif mode == "texte":
                display_arg((i+ajust)%N_tr,df,pred,col=right_column)
            elif mode == "probas":
                display_likelihood((i+ajust)%N_tr,like,col=right_column)
            elif mode == "pastilles":
                display_pastilles((i+ajust)%N_tr,sv,n=3)
            elif mode == "simil2":
                display_nearest((i+ajust)%N_tr,near,df.loc['moy'],col=right_column)
            elif mode == "simil1":
                display_nearest((i+ajust)%N_tr,near1,df.loc['moy'],col=right_column)
            elif mode == 'rule':
                display_rule((i+ajust)%N_tr,df,pred,col=right_column)
            elif mode == 'lime-line':
                display_shap((i+ajust)%N_tr,lime,col=right_column,plot='line')
            elif mode == 'lime-flot':
                display_shap((i+ajust)%N_tr,lime,col=right_column,plot='flot')
            elif mode == "lime-bar":
                display_shap((i+ajust)%N_tr,lime,col=right_column)
            elif mode == 'lime-pastille':
                display_pastilles((i+ajust)%N_tr,lime,n=4)
            elif mode == 'fake_rule':
                display_rule((i+ajust)%N_tr,df,pred,col=right_column,fake=True)
            display_score((i+ajust)%N_tr,pred,col=right_column,wording=2)
        winner = 'rouge' if labels.iloc[(i+ajust-1)%N_tr] else 'bleue'
        st.title("Équipe gagnante : " + winner)
        title = "ce slider ne prends pas de décision"
        st.select_slider(title,options_ope,"je ne sais pas",key="slide")
        st.button("Suite>>>",on_click=nextpage)
    
    display_match((i+ajust-1)%N_tr,df,col=left_column)
    
    st.sidebar.table(df_pres)
    
    
if st.session_state.cur_page == 1:
        
        # Pour la joie d'avoir l'eval.
        st.session_state['train'] = False # muda
        
        left_column, right_column = st.columns(2) # ou 3 ?
        

        i = left_column.selectbox("Choisissez le match à afficher",
                         list(range(1,N_tr+1)))
        
        with right_column:
            
            ajust = 0
            mode = "neutre"
            if st.session_state.aide:
                mode = st.radio("Mode de l'interface", ["confiance","simil",
                                         "rule","lime-pastille"])
            else:
                st.session_state.aide = st.button("Recommandation IA",
                                                  on_click=lambda: True)
            
            winner = 'rouge' if labels.iloc[(i+ajust-1)%N_tr] else 'bleue'
            st.title("Équipe gagnante : " + winner)
            # predi = st.session_state.reponses_num.loc[i+4,'rep']
            # st.title("Votre pronostic : " + predi)
            if mode != "neutre":
                display_score((i+ajust)%N_tr,pred,col=right_column,wording=2)
                
                if mode == "bar-SHAP":
                    display_shap((i+ajust)%N_tr,sv,col=right_column)
                elif mode == "texte":
                    display_arg((i+ajust)%N_tr,df,pred,col=right_column)
                elif mode == "probas":
                    display_likelihood((i+ajust)%N_tr,like,col=right_column)
                elif mode == "pastilles":
                    display_pastilles((i+ajust)%N_tr,sv,n=4)
                elif mode == "simil":
                    display_nearest((i+ajust)%N_tr,near,df.loc['moy'],col=right_column)
                elif mode == 'rule':
                    display_rule((i+ajust)%N_tr,df,pred,col=right_column)
                elif mode == "lime":
                    display_shap((i+ajust)%N_tr,lime,col=right_column)
                elif mode == 'lime-pastille':
                    display_pastilles((i+ajust)%N_tr,lime,n=4)#,col=right_column)
            
            st.button("Suite>>>",on_click=nextpage)
        
        display_match((i+ajust-1)%N_tr,df,col=left_column)
        title = "ce slider ne prends pas de décision"
        left_column.select_slider(title,options_ope,"je ne sais pas",key="slide")
        st.sidebar.table(df_pres)

if st.session_state.cur_page == 2:
        
        left_column, right_column = st.columns(2) # ou 3 ?
        

        i = left_column.selectbox("Choisissez le match à afficher",
                         list(range(1,N_tr+1)))
        
        with right_column:
            
            ajust = 0
            mode = st.selectbox("Mode de l'interface", ["neutre","bar-SHAP",
                                         "texte","probas","pastilles","simil",
                                         "rule","lime","lime-pastille"])
            
            winner = 'rouge' if labels.iloc[(i+ajust-1)%N_tr] else 'bleue'
            st.title("Équipe gagnante : " + winner)
            # predi = st.session_state.reponses_num.loc[i+4,'rep']
            # st.title("Votre pronostic : " + predi)
            if mode != "neutre":
                display_score((i+ajust)%N_tr,pred,col=right_column,wording=2)
                
                if mode == "bar-SHAP":
                    display_shap((i+ajust)%N_tr,sv,col=right_column)
                elif mode == "texte":
                    display_arg((i+ajust)%N_tr,df,pred,col=right_column)
                elif mode == "probas":
                    display_likelihood((i+ajust)%N_tr,like,col=right_column)
                elif mode == "pastilles":
                    display_pastilles((i+ajust)%N_tr,sv,n=4)
                elif mode == "simil":
                    display_nearest((i+ajust)%N_tr,near,df.loc['moy'],col=right_column)
                elif mode == 'rule':
                    display_rule((i+ajust)%N_tr,df,pred,col=right_column)
                elif mode == "lime":
                    display_shap((i+ajust)%N_tr,lime,col=right_column)
                elif mode == 'lime-pastille':
                    display_pastilles((i+ajust)%N_tr,lime,n=4)#,col=right_column)
            title = "ce slider ne prends pas de décision"
            st.select_slider(title,options_ope,"je ne sais pas",key="slide")
            st.button("Suite>>>",on_click=nextpage)
        
        display_match((i+ajust-1)%N_tr,df,col=left_column)
        
        st.sidebar.table(df_pres)
 


if st.session_state.cur_page == 1.1:
    
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

elif st.session_state.cur_page == 2.1:
    
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
    display_match(1,df,col=right_column)
        
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
    # df.to_csv("results/reponses_matchs_tr_p" + str(st.session_state.part)
    #           + '_' +  str(USER) + ".csv", index=False)
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
    # df.to_csv("results/reponses_matchs_ev_p" + str(st.session_state.part)
    #           + '_' +  str(USER) + ".csv", index=False)
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
    
    # with open("results/times_tr_" + str(USER) + ".pkl",'wb') as f:
        # pickle.dump(st.session_state.click_list,f)
    
    # with open("results/times_" + str(st.session_state.user) + ".pkl",'wb') as f:
    #     # pickle.dump(st.session_state.click_list,f)
    
    # on reindexe par les questions de formulaire.
    st.session_state['reponses_form'] = \
        st.session_state.reponses_form.reset_index(drop=True).rename(
            index=lambda x:  (list(range(12))+q1+q2)[x])
    
    # st.session_state.reponses_form.to_csv(
    #     "results/reponses_formulaire_" + str(st.session_state.user) + ".csv")

    
elif st.session_state.cur_page == 999:
    
    "Perfectly balanced..."
    "As everything should be."
    


if st.session_state.t_print is None: # permet le calcul du temps d'affichage.
    st.session_state['t_print'] = time.time()
st.write("page : "+str(st.session_state.cur_page))
clickable = True

