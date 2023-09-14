#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:05:25 2022

Ce fichier est un brouillon pour la conception du prochain main, correspondant
à la phase une 1 du protocole, l'acquisition de la performance humaine.

Il doit être éxécuté après training, et les deux codes vont être remis en un 
seul script normalement.

@author: Corentin.Boidot
"""

import streamlit as st
import numpy as np
import pandas as pd

import pickle
import time

# from functools import reduce


st.set_page_config(layout="wide")

pd.set_option("display.max_columns",40)
pd.set_option("display.max_rows",30)


USER = 1

#### Parameters

path = "" #"../Exp1-Streamlit/"


if "cur_page" not in st.session_state:
    st.session_state["cur_page"] = 0

if "cur_match" not in st.session_state:
    st.session_state["cur_match"] = 0

# i = st.session_state.cur_match -4

if "reponses_num" not in st.session_state:
    st.session_state["reponses_num"] = pd.DataFrame(np.zeros((0,3)),
                            columns=["rep","time_print","time_click"])
    
if "click_list" not in st.session_state:
    st.session_state["click_list"] = []    

if "t_print" not in st.session_state:
    st.session_state["t_print"] = None
    
# if "reponses_form" not in st.session_state:
#     st.session_state["reponses_form"] = pd.DataFrame(
#                                             pd.Series(np.zeros((0))),
#                                             columns=["rep"])

clickable = False # clickable = True en dernière ligne !

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

# "À quel fréquence faites vous X ?"
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



r1 = None

### Chargement des données ###

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

@st.cache
def init_data():
    '''
    df_select : pd.DataFrame, passé au new_order
    '''
    N = 40 # index à partir duquel on insère 
    with open(path+"selection_ev.pkl",'rb') as pf: 
        df_select = pickle.load(pf)

    df_select = df_select[new_order]
### traitement de la répétition
    # en maquette, je vais les répéter dans la deuxième dizaine.
    to_insert = df_select.iloc[:10].sample(5,random_state=USER)
    with open(path+"results/redondance_"+str(USER)+".pkl",'wb') as f:
        pickle.dump(to_insert,f)
    df_bis = df_select.iloc[:N].reset_index(drop=True)
    for i in range(5):
        df_bis.loc[N+2*i] = df_select.iloc[10+i] 
        df_bis.loc[N+2*i+1] = to_insert.iloc[i]
    df_bis = pd.concat([df_bis,df_select.reset_index(drop=True).iloc[N+5:50]]) \
        .reset_index(drop=True)
    df_bis.loc['moy'] = df_select.loc['moy']
    return df_bis
df_select = init_data()


### fonctions associées aux boutons "next" des différentes pages ###

def nextpage():
    global clickable
    if not clickable:
        # print("Ne cliquez pas si vite !")
        time.sleep(1)
    else:
        if st.session_state.cur_page in [3,4]:
            k = st.session_state.reponses_form.shape[0]
            st.session_state['reponses_form'].loc[k] = r1
            st.session_state['reponses_form'].loc[k+1] = r2
            st.session_state['reponses_form'].loc[k+2] = r3
        elif st.session_state.cur_page in [2]:
            st.session_state['reponses_form'] = \
                pd.DataFrame(np.array(r1).T,columns=["rep"])
        st.session_state.cur_page += 1
        st.session_state.click_list.append(time.time())        
        st.session_state['t_print'] = None
        st.session_state["completed"] = [False]*5
        clickable = False
        time.sleep(.3)
        
def nextexample():
    global clickable
    if not clickable:
        # print("On ne vous a pas appris que le double-click, c'est mal ?")
        time.sleep(1)
    else:
        st.session_state.cur_match += 1
        line = [r1, st.session_state.t_print, time.time()]
        st.session_state['reponses_num'].loc[st.session_state.cur_match-1] = line
        st.session_state['t_print'] = None
        clickable = False
        # st.session_state["slide"] = "je ne sais pas"
        time.sleep(.3)
        if st.session_state.cur_match == 55:
            clickable = True # sinon la ligne suivante est bloquée
            nextpage()

j = st.session_state.cur_page

# 'recense le premier mort de la partie (1 s’il est de l’équipe bleue, 0 s’il est de l’équipe rouge).'
@st.cache
def make_df_pres():
    l_ter = ["recense le premier mort de la partie ('blue' = bleue, 'red' = rouge).",
     'nombre de dragons (grand monstre) détruits par l’équipe bleue.',
     'nombre de héraults (grand monstre) détruits par l’équipe bleue.',
     'nombre de tourelles détruites par l’équipe bleue.',
     'nombre de dragons (grand monstre) détruits par l’équipe rouge.',
     'nombre de héraults (grand monstre) détruits par l’équipe rouge.',
     'nombre de tourelles détruites par l’équipe rouge.',
     'nombre de balises de vision placées par l’équipe bleue.',
     'nombre de balises ennemies détruites par l’équipe bleue.',
     'nombre de champions adverses tués par les champions bleus.',
     'nombre d’assists (coopérations de champions à un kill) au sein de l’équipe bleue.',
     'quantité totale de pièces d’or des champions bleus.',
     'quantité totale d’expérience des champions bleus.',
     'nombre de sbires ennemis détruits par les chamions bleus.',
     'nombre de petits monstres de la jungle détruits par les champions bleus.',
     'nombre de balises de vision placées par l’équipe rouge.',
     'nombre de balises ennemies détruites par l’équipe rouge.',
     'nombre de champions adverses tués par les champions rouges.',
     'nombre d’assists (coopérations de champions à un kill) au sein de l’équipe rouge.',
     'quantité totale de pièces d’or des champions rouges.',
     'quantité totale d’expérience des champions rouges.',
     'nombre de sbires ennemis détruits par les chamions rouges.',
     'nombre de petits monstres de la jungle détruits par les champions rouges.']
    keys = old_order
    dic = {"features_names":keys,"description":l_ter}
    df_pres = pd.DataFrame(dic)
    df_pres = df_pres.set_index("features_names")
    df_pres = df_pres.T[new_order].T #cassage de crâne
    return df_pres
df_pres = make_df_pres()

options_ope = ["l’équipe bleue va certainement gagner",
            "l’équipe bleue va vraisemblablement gagner",
            "l’équipe bleue a un léger avantage sur l’équipe rouge",
            "je ne sais pas",
            "l’équipe rouge a un léger avantage sur l’équipe bleue",
            "l’équipe rouge va vraisemblablement gagner",
            "l’équipe rouge va certainement gagner"]

nondutout = ["Non pas du tout", "non", "plutôt non", "indécis", "plutôt oui", "oui", "oui tout à fait"]
pasdaccord = ["Pas du tout d’accord", "pas d’accord", "plutôt pas d’accord", "neutre", "plutôt d’accord", "d’accord", "tout à fait d’accord"]



##############################################################################
#               ICI COMMENCE L'INTERACTION AVEC L'UTILISATEUR                #
##############################################################################


if st.session_state.cur_page == 0:
    ###Présentation ?
    st.title("Début du test")
    
    "Objectif : prédire l'équipe gagnante d'un match de League of Legends à partir de données statistiques à 10 minutes du match."
    
    st.button("START>>>",key='nekst',on_click=nextpage)
    
elif st.session_state.cur_page == 1:
    
    left_column, right_column = st.columns(2)
    
    
    with right_column:
        
        i = st.session_state.cur_match
        st.title("Match n°"+str(i+1))
        
        title = 'Quel est votre pronostic ?'
        
        r1 = st.select_slider(title,options_ope,"je ne sais pas",key="slide")

        increment = st.button("NEXT>>>",on_click=nextexample)
    
    tab = df_select.iloc[[i,-1]].T
    tab.columns = ["match","moy"]
    left_column.table(tab.astype(str)) 
    
    st.sidebar.table(df_pres)

elif st.session_state.cur_page == 2:
    
    st.title("Questions sur votre expérience avec League of Legends")
    
    r1 = [0]*9
    
    l="À quelle fréquence jouez-vous à League of Legends ?"
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
    
    l="Avez-vous des connaissances en science des données ? (choisissez au plus proche)"
    r1[8] = st.select_slider(l,datascience)
    
    # st.write(r1)
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    

elif st.session_state.cur_page == 3:
    
    st.title("Ressentis au cours de l'expérience")
    
    r1 = st.select_slider("Pensez-vous avoir progressé au cours de l’utilisation ?",
                          nondutout,"indécis")
    
    r2 = st.select_slider("Étiez-vous très concentré durant l’expérience ?",
                          nondutout,"indécis")
    
    r3 = st.select_slider("L’expérience vous a-t-elle fatigué ?",
                          nondutout,"indécis")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    
    
elif st.session_state.cur_page == 4:
    
    st.title("Informations générales")
    
    r1 = st.radio("Vous êtes : ", ["un homme","une femme","préfère ne pas répondre"])
    
    r2 = st.text_input("Quel âge avez-vous : ")
    
    r3 = st.text_input("Y a-t-il d'autres points à relever sur votre profil ou sur le déroulé de l'expérience ?")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    
    
elif st.session_state.cur_page == 5:
    
    st.title("Félicitation, vous avez fini !")
    
    
    # st.write(st.session_state.reponses_form)
    st.session_state.reponses_num.to_csv("results/reponses_matchs_ev_" 
                                         + str(USER) + ".csv")
    with open("results/times_ev_" + str(USER) + ".pkl",'wb') as f:
        pickle.dump(st.session_state.click_list,f)
    
    st.session_state.reponses_form.to_csv("results/reponses_formulaire_" 
                                         + str(USER) + ".csv")
    "Merci pour votre participation ! Vous pouvez arrêter le programme dans le terminal avec Ctrl+C, puis fermer l'onglet."
    

if st.session_state.t_print is None:
    st.session_state['t_print'] = time.time()
st.write("page : "+str(j))
clickable = True
