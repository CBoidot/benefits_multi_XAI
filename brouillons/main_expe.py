#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:29:34 2022

Version 1 du code achevée le 28 janvier 2022.

L'objectif de ce script d'implémenter un protocole d'expérience décrit ici : 
https://docs.google.com/presentation/d/1CBTa-vVZCyNd9Lqo8MzHpUe6sedvj2ytP7DQfCCjQSw/edit#slide=id.gc6f73a04f_0_0
Le but de l'expérience est d'évaluer l'utilité d'explications produites par la
librairie shap, dans le cadre d'une activité de pronostic sur des matchs d'e-
sport.

À utiliser avec l'environnement expenv1, stockée dans le même dossier.

J'ai un pseudo-historique des modificiations apportées "au fil de l'eau" (ie:
sans méthodo, au besoin).
v1: utilise les données "old", a eu besoin d'une rustine d'affichage pour que 
les bonnes données soient affichées sur le graphe de SHAP.
v1.1: modification des données, et de l'ordre des colonnes. 
v1.2: en fait, les SV du précédent avaient été calculées n'importe comment, ce
qui donnait toujours une sommes de SV au dessus de .5 (rouges qui gagnent 
toujours sur le graphique, donc).
v1.2.2: corrections graphiques (à nouveau l'affichage des données dans le 
graphique)
v1.3: correction du cur_match, qui nous faisait passer 2 fois le match 19

@author: Corentin.Boidot
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import shap
import time


st.set_page_config(layout="wide") # enfin !
## résout mes principaux problèmes avec l'affichage

# pd.set_option("display.precision", 2) #df.style.format("{:.2%}")
pd.set_option("display.max_columns",40)
pd.set_option("display.max_rows",30)


#### Parameters

path = "" #"../Exp1-Streamlit/"

if "user" not in st.session_state:
    st.session_state["user"] = input("N° de l'expérience : ")

if "cur_page" not in st.session_state:
    st.session_state["cur_page"] = 0

if "cur_match" not in st.session_state:
    st.session_state["cur_match"] = 0

if "sv_first" not in st.session_state:
    if int(st.session_state.user)%2 == 0:
        st.session_state["sv_first"] = False 
    else:
        st.session_state["sv_first"] = True

# if "with_sv" not in st.session_state:
    # st.session_state["with_sv"] = True # False # 
i = st.session_state.cur_match
if i in [10,11,12,13,14,15,16,17,18,19,25,26,27,28,29,35,36,37,38,39]:
    st.session_state["with_sv"] = True
else:
    st.session_state["with_sv"] = False
if st.session_state.sv_first and i<20:
    st.session_state["with_sv"] = not st.session_state["with_sv"]
        
if "transition" not in st.session_state: # pour ne pas modifier ma pagination
    st.session_state["transition"] = False
    
# if "ope_first" not in st.session_state:
#     st.session_state["ope_first"] = False

# if "ope" not in st.session_state:
#     st.session_state["ope"] = False
#=> je fais les choses de manière fixe, avec des copiés-collés

if "reponses_num" not in st.session_state:
    st.session_state["reponses_num"] = pd.DataFrame(np.zeros((0,3)),
                            columns=["rep","time_print","time_click"])
    # généralement, il n'y a pas un dixième 
    
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

options_ope = ["l’équipe bleue va certainement gagner",
            "l’équipe bleue va vraisemblablement gagner",
            "l’équipe bleue a un léger avantage sur l’équipe rouge",
            "je ne sais pas",
            "l’équipe rouge a un léger avantage sur l’équipe bleue",
            "l’équipe rouge va vraisemblablement gagner",
            "l’équipe rouge va certainement gagner"]

inverted_opt = np.flip(options_ope)

# opt_red = [options_ope[0]+" (le modèle se trompe)"]


### Chargement des données ###

old_order = ['firstBlood', 'blueDragons', 'blueHeralds', 'blueTowersDestroyed',
       'redDragons', 'redHeralds', 'redTowersDestroyed', 'blueWardsPlaced',
       'blueWardsDestroyed', 'blueKills', 'blueAssists', 'blueTotalGold',
       'blueTotalExperience', 'blueTotalMinionsKilled',
       'blueTotalJungleMinionsKilled', 'redWardsPlaced', 'redWardsDestroyed',
       'redKills', 'redAssists', 'redTotalGold', 'redTotalExperience',
       'redTotalMinionsKilled', 'redTotalJungleMinionsKilled']
# 'blueFirstBlood'
new_order = ['firstBlood', 'blueDragons', 'redDragons', 'blueHeralds',
 'redHeralds', 'blueTowersDestroyed', 'redTowersDestroyed', 'blueWardsPlaced',
 'redWardsPlaced', 'blueWardsDestroyed', 'redWardsDestroyed', 'blueKills', 
 'redKills', 'blueAssists', 'redAssists', 'blueTotalGold', 'redTotalGold',
 'blueTotalExperience', 'redTotalExperience', 'blueTotalMinionsKilled', 
 'redTotalMinionsKilled', 'blueTotalJungleMinionsKilled',
 'redTotalJungleMinionsKilled']

@st.cache
def init_sv():
    '''
    xpo : explication telle que définies dans shap. xpo.data est en old order
    df_select : pd.DataFrame, passé au new_order
    pred : np.array
    '''
    with open(path+"shapley_values.pkl",'rb') as pf:
        # "sv_vrac_RF.pkl" à l'époque des test dans ../These-1A/
        xpo = pickle.load(pf)
    with open(path+"means.pkl",'rb') as pf:
        m = pickle.load(pf)
        m = m.apply(lambda x: round(x,2) if str(x)!=x else x)
    # df_select = xpo.data  # remplaçait les lignes suivantes
    with open(path+"selection.pkl",'rb') as pf: 
        df_select = pickle.load(pf)
    df_select.loc["moy"] = m
    df_select.loc['moy','firstBlood'] = m.firstBlood
    with open(path+"predictions.pkl",'rb') as pf:
        # "decision.pkl" à l'époque des test dans ../These-1A/
        pred = pickle.load(pf)
    df_select = df_select[new_order]
    # xpo.data = df_select.drop('moy') # il y eut un problème : on reseta
    # print(xpo.values[0][19])
    return xpo, df_select, pred.values
xpo, df_select, pred = init_sv()



### fonctions associées aux boutons "next" des différentes pages ###

def nextpage():
    global clickable
    if not clickable:
        # print("Ne cliquez pas si vite !")
        time.sleep(1)
    else:
        if st.session_state.cur_page in [0,9,10,11,12]:
            # if r1 is None or r2 is None or r3 is None or r4 is None:
            #     return None
            k = st.session_state.reponses_form.shape[0]
            st.session_state['reponses_form'].loc[k] = r1
            st.session_state['reponses_form'].loc[k+1] = r2
            st.session_state['reponses_form'].loc[k+2] = r3
            st.session_state['reponses_form'].loc[k+3] = r4
            st.session_state['reponses_form'].loc[k+4] = r5
        elif st.session_state.cur_page in [13,14]:
            k = st.session_state.reponses_form.shape[0]
            st.session_state['reponses_form'].loc[k] = r1
            st.session_state['reponses_form'].loc[k+1] = r2
            st.session_state['reponses_form'].loc[k+2] = r3
        elif st.session_state.cur_page == 4:
            k = st.session_state.reponses_form.shape[0]
            st.session_state['reponses_form'].loc[k] = r1
        if st.session_state.cur_page in [3,4,]:
            st.session_state['transition'] = True
        st.session_state.cur_page += 1
        st.session_state.click_list.append(time.time())
        
        st.session_state['t_print'] = None
        clickable = False
        time.sleep(.3)
        
def nextexample():
    global clickable
    if not clickable:
        # print("On ne vous a pas appris que le double-click, c'est mal ?")
        time.sleep(1)
    else:
        st.session_state.cur_match += 1
        # print(r1)
        line = [r1, st.session_state.t_print, time.time()]
        # print(line)
        st.session_state['reponses_num'].loc[i] = line
        st.session_state['t_print'] = None
        clickable = False
        # st.session_state["slide"] = "je ne sais pas"
        time.sleep(.3)
        if st.session_state.cur_match in [10,25,30,35]:
            st.session_state.transition = True
        elif st.session_state.cur_match in [20,40]:
            clickable = True # sinon la ligne suivante est bloquée
            nextpage() # l'innovation de la v1.3.1 :smiley sunglass:
        
def samepage():
    global clickable
    if not clickable:
        # print("On ne vous a pas appris que le double-click, c'est mal ?")
        time.sleep(1)
    else:
        st.session_state.transition = False
        clickable = False
        st.session_state.click_list.append(time.time())
        st.session_state['t_print'] = None
        time.sleep(.3)
    
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

nondutout = ["Non pas du tout", "non", "plutôt non", "indécis", "plutôt oui", "oui", "oui tout à fait"]
pasdaccord = ["Pas du tout d’accord", "pas d’accord", "plutôt pas d’accord", "neutre", "plutôt d’accord", "d’accord", "tout à fait d’accord"]


### Données d'exemple

# with open("joke_example.pkl",'rb') as f:
#     xpo_joke = pickle.load(f)
# data = pd.DataFrame(xpo_joke.data,
#                     columns=["blueTeemo","redTeemo","blueMagicDmg","bluePhysicalDmg","redMagicDmg","redPhysicalDmg"])
# data.loc["moy"] = [.03,.03,120.01,143.27,120.00,143.29]
# xpo_joke.values = np.array([.02,.45,.07,.04,-.07,-.05])


# Rustines de la v1.3



##############################################################################
#               ICI COMMENCE L'INTERACTION AVEC L'UTILISATEUR                #
##############################################################################


# Transition entre les matchs avec ou sans valeurs de Shapley

if st.session_state.transition:
    
    if st.session_state.cur_match != 0:
        st.title("Transition")
        if st.session_state.with_sv:
            "Pour les prochains matchs, vous aurez accès à l'interface d'analyse du modèle."
        else:
            "Pour les prochains matchs, vous n'aurez pas accès à l'interface d'analyse du modèle."
    
    elif j==4:
        ## premier affichage, sans SV
        with open("example_data.pkl",'rb') as pf:
            extrait = pickle.load(pf)
            extrait = extrait[new_order]
        with open("example_score.pkl",'rb') as pf:
            y_p_ex = pickle.load(pf)
        st.title("Utilisation de l'interface")
    
        left_column, right_column = st.columns(2) # ou 3 ?
        
        with right_column:
            "*\u2193 En dessous, vous avez la recommandation d'équipe sur laquelle parier du modèle, avec sa probabilité.*"
            st.title("**Victoire prédite : équipe **rouge")
            st.title("**Probabilité estimée :**"+str(y_p_ex.iloc[0])+"%")
            
            "*L'outil de sélection ci-dessous vous permettras d'exprimer votre pronostic, sur une échelle à 7 crans.* \u2193"
            r1 = st.select_slider("L'utilisation de cette interface me permettras de prendre une décision.",
                                  pasdaccord,"neutre")
            st.button("NEXT>>>",on_click=samepage)
            
        left_column.write("*En dessous, les valeurs pour le match en cours, avec les moyennes* \u2193")
        
        data = pd.concat([extrait.iloc[[0]],df_select.loc[["moy"]]],axis=0)
        tab = data.T
        tab.columns = ["match exemple","moy"]
        left_column.table(tab.astype(str))#.style.format("{:.2f}"))
        left_column.write("Ce match a effectivement été gagné par l'équipe rouge.")
        
    elif j==5:
        st.title("Début de l'expérience")
        "*Cliquez pour commencer.*"
    
    else:
        "Erreur code : allez chercher Corentin Boidot."
    
    if j!=4:
        st.button("NEXT>>>",key='nekst',on_click=samepage)

elif st.session_state.cur_page == 0:
    
    ### Formulaire 
    
    st.title("Bonjour.")
    
    # filled = True
    r1 = st.radio("Connaissez-vous un peu le jeu League of Legends ?",
                  ["non","oui"])
    r2 = st.radio("Avez-vous déjà joué ou regardé une partie en entier ?",
                  ["non","oui"])
    r3 = st.radio("Jouez-vous régulièrement à des MOBAs ?",
                  ["non","oui"])
    r4 = st.select_slider("Pensez-vous pouvoir faire de bons pronostics sur la victoire au bout de 10 minutes de jeu ?",
                nondutout,"indécis")
    r5 = st.text_input("(Si vous jouez en classé) quel est votre rang en classé ?")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)#,disabled=filled)
    
elif st.session_state.cur_page == 1:
    
    ### Description de League of Legend
    
    st.title("Description de League of Legend")
    "*À lire pour ceux qui ne connaissent pas le jeu !*\n"
    
    "League of Legends (LoL) est un jeu de stratégie en équipe, de type MOBA. L'objectif de chaque équipe consiste à détruire le nexus dans la base ennemie (son bâtiment central), avant que l’équipe adverse ne détruise le sien. Avant le lancement de la partie, chaque joueur choisit un champion dans une liste, chacun ayant ses atouts particuliers (combattant à distance, en  mêlée ou encore mages lançant des sorts pour affaiblir ou blesser l'ennemi)."
    "Au fil de la partie les champions deviennent plus puissants de deux façons : le gain de “niveau” obtenus en gagnant de **l'expérience** ; un champion gagne de l’expérience quand un ennemi est éliminé à proximité. Le second moyen de gagner en puissance consiste à acheter des *objets* grâce à la boutique dans la base de chaque équipe, avec les **pièces d'or** obtenues en tuant les ennemis."
    "L’équipe apparaît dans sa base, et chaque champion y réapparaît après avoir été tué. Partant de la base, les champions se répartissent sur 3 voies (ou “lanes”), celle du bas (bottom), du milieu (mid) et celle du haut (top). Entre ces voies se trouve les zones de jungle, habitées par des monstres puissants (les dragons et les hérauts) capables de tuer un champion mal préparé. Tuer ces créatures  donne de l'or, de l'expérience et des bonus temporaires."
    "Chaque voie relie les deux bases et est protégée par des **tourelles** : il faut détruire les tourelles pour assiéger la base adverse et détruire son nexus. Ces tourelles peuvent facilement tuer un champion. Pour les détruire sans risque, les joueurs rester derrière les sbires de leur équipe."
    "Les **sbires** (minions) sont des unités contrôlés par l'ia qui progressent le long des voies et qui attaquent les ennemis en chemin : chaque équipe a ses sbires, et ils sont des proies relativement faciles pour les champions adverses qui gagnent ainsi de l'expérience et des pièces d'or. Quand les sbires et leur champion atteignent une tourelle adverse, le champion peut attaquer la tourelle sans risque pendant qu’elle attaque les sbires, au moins tant que les champions adverses sont absents."
    "Quand un champion meurt, son joueur doit patienter avant qu’il ne réapparaisse à sa base. Il faut attendre de plus en plus longtemps pour réapparaître : de quelques secondes en début de partie à près d'une trentaine vers la fin. Le champion ne perd ni or, ni équipement, ni expérience quand il meurt."
    #"Ce sont les bases de League of Legends : accumuler de l’expérience et de l'or, acheter des objets et détruire la base adverse tout en affrontant des adversaires qui ont le même objectif."
    "Le jeu est très connu, notamment pour son aspect compétitif qui l’a mis au devant de la scène de l’e-sport. Les joueurs peuvent être classé en lignes, dans neuf ligues, Fer, Bronze, Argent, Or, Platine, Diamant, Maître, Grand Maître et Challenger. Le jeu est régulièrement mis à jour, et le ‘metagame’ évolue avec les saisons."
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    

elif st.session_state.cur_page == 2:
    
    ### Présentation du problème
    
    st.title("Présentation de votre mission")
    "***À lire impérativement par tous.***\n"
    
    "On tente de faire des pronostics de League of Legends (prédire quelle équipe va à gagner). On prend les paris à partir de statistiques de jeu prises 10 minutes après le départ. Il s’agit de matchs classés solo de rang Diamant I à Master, durant la saison 10 de LoL (les joueurs ne se connaissent pas avant le début du match : les équipes sont formées de façon impartiales)."
    "Un modèle algorithmique a été développé pour vous fournir une aide au pronostic sous forme d’un score (la probabilité de victoire estimée de chaque équipe). L’algorithme n’est pas assez performant pour être utilisé seul : un humain est nécessaire pour prendre la décision."
    "Vous aurez donc une suite de match à évaluer, sous la forme de relevés de données de jeu, et vous disposerez du score produit par l’algorithme. Selon les matchs, vous aurez également accès à une interface d’analyse du modèle pour vous aider. Votre évaluation revient à une décision (une sorte de pari) que vous seriez prêt à prendre vis-à-vis du match."
    "Le format des données est présenté à la page suivante."
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    

elif st.session_state.cur_page == 3:
    
    ### Présentation des données
    st.title("Présentation des données statistiques")
    
    st.table(df_pres)
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)


elif st.session_state.cur_page == 4:
    
    ### Utiliser l'interface
    st.title("Utilisation de l'interface")
    
    st.sidebar.table(df_pres) 
    left_column, right_column = st.columns(2) # ou 3 ?
    
    with open("example_data.pkl",'rb') as pf:
        extrait = pickle.load(pf)[new_order]
    with open("example_score.pkl",'rb') as pf:
        y_p_ex = pickle.load(pf)
    with open("example_sv.pkl",'rb') as f:
        xpo_ex = pickle.load(f)
        xpo_ex.data = extrait[old_order][1:]
    
    with right_column:
        st.title("**Victoire prédite : équipe **bleue")
        st.title("**Probabilité estimée :**"+str(int((1-y_p_ex.iloc[1])*100))+"%")
        "*Ci-dessous, l'interface d'analyse du modèle, qui sera masquée pour la moitié des matchs. \nLa longueur des barres est proportionnelle à l'importance d'une caractéristique statistique dans la prédiction du modèle. Vers la gauche, les caractéristiques jouant pour l'équipe bleue, à droite pour l'équipe rouge.* \u2193"
        shap.plots.bar(xpo_ex[0],show_data=True,show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        r1 = st.select_slider("L'utilisation de cette interface me permettras de prendre une décision.",
                              pasdaccord,"neutre")
        st.button("NEXT>>>",key='nekst',on_click=nextpage)
    left_column.write("\u2190 *Vous pouvez masquer/afficher le rappel des données.*")
    # left_column.write("*En dessous, les valeurs pour le match en cours, avec les moyennes* \u2193")
    
    data = pd.concat([extrait.iloc[[1]],df_select.loc[["moy"]]])
    tab = data.T
    tab.columns = ["match exemple 2","moy"]
    left_column.table(tab.astype(str))
    left_column.write("Ce match a effectivement été gagné par l'équipe bleue.")
    
    # "***Les données utilisées ici sont des données fictives : elles ne sont pas au format de la suite de l'expérience.***"
    

elif st.session_state.cur_page == 5:
    
    ### Série 1 : opérateur, pronostic sur un match
    
    i = st.session_state["cur_match"]
    left_column, right_column = st.columns(2) # ou 3 ?
    with right_column:
        # st.write("Probabilité de victiore estimée de l'équipe rouge : "
        #           + str(pred[i]*100)+"%")
        if pred[i]>.5:
            st.title("**Victoire prédite : équipe** rouge")
            st.title("**Probabilité estimée :** " + str(int(pred[i]*100))+"%")
        else:
            st.title("**Victoire prédite : équipe** bleue")
            st.title("**Probabilité estimée :** "+str(int((1-pred[i])*100))+"%")
        if st.session_state.with_sv:
            shap.plots.bar(xpo[st.session_state.cur_match],show_data=True,show=False)
            fig = plt.gcf()
            st.pyplot(fig)
        title = 'Quel est votre pronostic ?'
        r1 = st.select_slider(title,options_ope,"je ne sais pas",key="slide")#,on_change=Callb)
        # if st.session_state.cur_match < 19:
        increment = st.button("NEXT>>>",on_click=nextexample)
        # else:
        #     increment = st.button("NEXT>>>",on_click=nextpage)
        # st.write("page : "+str(i))
    
    tab = df_select.iloc[[i,-1]].T
    # tab.iloc[1:] = tab.iloc[1:].apply(lambda x: round(x,2)).T
    # tab.iloc[0:1] = tab.iloc[0:1].style
    tab.columns = ["match","moy"]
    left_column.title("Match n°"+str(i))
    left_column.table(tab.astype(str)) # .style.format("{:.2f}") # table # dataframe
    
    st.sidebar.table(df_pres)
    
    # i = st.session_state["cur_page"]


elif st.session_state.cur_page == 6:
    
    st.title("Seconde séquence")
    
    "Changement de contexte : puisque vous testez l'algorithme, on vous invite à participer à l'évaluation du modèle, afin d'aider ses concepteurs à trouver de potentiels bugs à réparer."
    "Votre mission diffère seulement légèrement d'avant : votre avis est donné par rapport à l'algorithme."
    "/!\\ *Votre réponse doit être donnée par rapport au choix équipe rouge/bleue, et non par rapport au niveau de confiance affiché.* /!\\"
    
    st.session_state['cur_match'] = 20 # V1.3, le 8/02/2022
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)


elif st.session_state.cur_page == 7:


    ### Série 2 : évaluateur, décision relative au modèle
    
    i = st.session_state["cur_match"]
    left_column, right_column = st.columns(2) # ou 3 ?
    with right_column:
        if pred[i]>.5:
            st.title("**Victoire prédite : équipe** rouge")
            st.title("**Probabilité estimée :** " + str(int(pred[i]*100))+"%")
        else:
            st.title("**Victoire prédite : équipe** bleue")
            st.title("**Probabilité estimée :** "+str(int((1-pred[i])*100))+"%")
        if st.session_state.with_sv:
            shap.plots.bar(xpo[st.session_state.cur_match],
                           show_data=True,show=False)
            fig = plt.gcf()
            st.pyplot(fig)
        team = "rouge" if pred[i]>.5 else "bleue"
        title = 'Pensez-vous que le modèle parie sur la bonne équipe ('+team+")"
        # if pred[i]>.5:
            # r1 = st.select_slider(title,options_ope,"je ne sais pas",key="slide")
        # else:
            # r1 = st.select_slider(title,inverted_opt,"je ne sais pas",key="slide")
        r1 = st.radio(title,np.flip(nondutout))

        increment = st.button("NEXT>>>",key='ex',on_click=nextexample)
        
    
    tab = df_select.iloc[[i,-1]].T
    tab.columns = ["match","moy"]
    left_column.title("Match n°"+str(i))
    left_column.table(tab.astype(str)) 
    st.sidebar.table(df_pres)


elif st.session_state.cur_page == 8:
    
    st.title("Félicitation !")
    
    "Vous avez terminé l'analyse de match."
    "Avant de finir, merci de répondre à quelques questions :"
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    

elif st.session_state.cur_page == 9:
    
    st.title("Questions sur l'interface d'analyse du modèle informatique")
    "Êtes-vous d'accord avec les propositions suivantes :"
    
    r1 = st.select_slider("L’interface d'analyse comprenait toutes les informations pertinentes pour m’aider à prendre une décision.",
                          pasdaccord,"neutre")
    r2 = st.select_slider("L’interface d'analyse m’a permis de prendre une décision plus rapide.",
                          pasdaccord,"neutre")
    r3 = st.select_slider("L’interface d'analyse m’a été utile pour faire un pronostic.",
                          pasdaccord,"neutre")
    r4 = st.select_slider("L’interface était facile à utiliser.",
                          pasdaccord,"neutre")
    r5 = st.text_input("Comment c’est passé votre expérience avec l’interface d'analyse du modèle ? Comment l’avez-vous utilisée ?")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    
    
elif st.session_state.cur_page == 10:
    
    st.title("Questions sur l'algorithme")
    "L'algorithme d'aide au pronostic utiliserait correspond à ce que l'on appelle communément un modèle d'intelligence artificielle (IA)."
    "Exprimez votre accord avec les propositions suivantes :"
    
    r1 = st.select_slider("L’interface d'analyse m’a permis de saisir comment l’IA fonctionnait.",
                           pasdaccord,"neutre")
    
    r2 = st.select_slider("L’IA utilisée est capable de faire de proposer de bons pronostics.",
         pasdaccord,"neutre")
    
    r3 = st.select_slider("L’interface d’analyse expliquait bien le modèle, de façon claire et concise.",
                          pasdaccord,"neutre")
    
    r4 = st.text_input("Qu’attendriez-vous d’une IA qui tente de vous expliquer sa décision ?")
    
    r5 = st.select_slider("Si vous faisiez réellement des pronostics à 10 minutes, aimeriez-vous avoir un modèle pour vous assister ?",
                          nondutout, 'indécis')
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    

elif st.session_state.cur_page == 11:
    
    st.title("Utilisation de l'IA et d'interfaces explicatives")
    
    r1 = st.select_slider("Si vous faisiez réellement des pronostics à 10 minutes, aimeriez-vous avoir accès aux données moyennes ?",
                          nondutout, 'indécis')
    
    r2 = st.select_slider("Si vous faisiez réellement des pronostics à 10 minutes, aimeriez-vous avoir une interface d'analyse de modèle ?",
                          nondutout, "indécis")
    
    r3 = st.select_slider("Avez confiance en le développement futur de l'IA ?",
                          nondutout,'indécis')
    
    r4 = st.select_slider("Seriez-vous prêt à utiliser un système d’IA similaire, avec xp° des résultats, dans un autre contexte ?",
                          nondutout,'indécis')
    
    r5 = st.text_input("Quelles sont vos attentes vis-à-vis de l’utilisation de l’IA dans un cadre d’application similaire ?")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    

elif st.session_state.cur_page == 12:
    
    st.title("Expertise en IA")
    
    r1 = st.radio(" Quel est votre niveau de formation en informatique / sciences de l’ingénieur ?",
                          ["Aucun", "Bac spé math/info", "Bac+2", "Bac+3 (équivalent licence)", "Bac+4", "Bac+5 (équivalent master)", "Bac+6", "supérieur"])
    
    r2 = st.radio("Avez-vous des connaissances en Intelligence Artificielle ?",
                  ["Aucune", "en amateur (par la vulgarisation)", "passionné d’informatique et de statistique (connaissances techniques)", "jeune pro (M2 ou doctorant)", "confirmé (plus de trois ans de travail dans le domaine du Machine Learning"])
    
    r3 = st.text_input("Avez-vous d’autres connaissances liées à l’AI ou l’XAI en particulier ?")
                       
    r4 = st.select_slider("Distinguez-vous intuitivement les trois fonctions suivantes : la probabilité des données, la probabilité de victoire de l’équipe rouge, et la fonction calculée par le modèle de prédiction ?",
                          nondutout, "indécis")
    
    r5 = st.select_slider("Pensez-vous être capable d’estimer la probabilité de victoire de l’équipe rouge ?",
                          nondutout,"indécis")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)


elif st.session_state.cur_page == 13:
    
    st.title("Ressentis au cours de l'expérience")
    
    r1 = st.select_slider("Pensez-vous avoir progressé au cours de l’utilisation ?",
                          nondutout,"indécis")
    
    r2 = st.select_slider("Étiez-vous très concentré durant l’expérience ?",
                          nondutout,"indécis")
    
    r3 = st.select_slider("L’expérience vous a-t-elle fatigué ?",
                          nondutout,"indécis")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    
    
elif st.session_state.cur_page == 14:
    
    st.title("Informations générales")
    
    r1 = st.radio("Vous êtes : ", ["un homme","une femme","préfère ne pas répondre"])
    
    r2 = st.text_input("Quel âge avez-vous : ")
    
    r3 = st.text_input("Y a-t-il d'autres points à relever sur votre profil ou sur le déroulé de l'expérience ?")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    
    
elif st.session_state.cur_page == 15:
    
    st.title("Félicitation, vous avez fini !")
    
    "\nPour de vrai, cette fois. ;)"
    
    st.session_state.reponses_num.to_csv("results/reponses_matchs_" 
                                         + st.session_state.user + ".csv")
    with open("results/times_" + st.session_state.user + ".pkl",'wb') as f:
        pickle.dump(st.session_state.click_list,f)
    
    st.session_state.reponses_form.to_csv("results/reponses_formulaire_" 
                                         + st.session_state.user + ".csv")







if j not in [5,7]:
    st.write("page : "+str(j))
else:
    st.write("page : "+str(j)+"."+str(i%20))
    
if st.session_state.t_print is None:
    st.session_state['t_print'] = time.time()

clickable = True

# print("c'est bon")





## /!\ Il est très difficile de construire une xp° à la main
# je m'y suis piqué sur mon exemple...

# Voici un chemin :
# lr = LogisticRegression()

# data = pd.DataFrame([[0,2,0,0,0,0]],
#     columns=["blueTeemo","redTeemo","blueMagicDmg","bluePhysicalDmg","redMagicDmg","redPhysicalDmg"])
# data.loc["moy"] = [.03,.03,120.01,143.27,120.00,143.29]
# lr.fit(data,np.array([1,0]))
# xpr = shap.Explainer(lr,data)
# xpo_ = xpr(data.iloc[0:1])
# xpo_.values = np.array([[.02,.48,.07,.04,-.07,-.05]])
# shap.plots.bar(xpo_[0],show_data=True)
# with open("../Exp1-Streamlit/joke_example.pkl",'wb') as pf:
#      pickle.dump(xpo_,pf)


# Anciennement en page 4

#     st.sidebar.table(df_pres) 
#     left_column, right_column = st.columns(2) # ou 3 ?
#     with right_column:
#         "*En dessous, vous avez avez la recommandation d'équipe sur laquelle parier du modèle, avec sa probabilité.*"
#         st.title("**Victoire prédite : équipe **rouge")
#         st.title("**Probabilité estimée :** 99%")
#         "*Ci-dessous, l'interface d'analyse du modèle, qui sera masquée pour la moitié des matchs. \nLa longueur des barres est proportionnelle à l'importance d'une caractéristique statistique dans la prédiction du modèle. Vers la gauche, les caractéristiques jouant pour l'équipe bleue, à droite pour l'équipe rouge.* \u2193"
#         shap.plots.bar(xpo_joke[0],show_data=True,show=False)
#         fig = plt.gcf()
#         st.pyplot(fig)
#         "*L'outil de sélection ci-dessous vous permettras d'exprimer votre pronostic, sur une échelle à 7 crans.* \u2193"
#         r1 = st.select_slider("L'utilisation de cette interface me permettras de prendre une décision.",
#                               pasdaccord,"neutre")
#         st.button("NEXT>>>",key='nekst',on_click=nextpage)
#     left_column.write("\u2190 *Vous pouvez masquer/afficher le rappel des données.*")
#     left_column.write("*En dessous, les valeurs pour le match en cours, avec les moyennes* \u2193")
    
#     tab = data.T
#     tab.columns = ["match fictif","moy"]
#     left_column.table(tab.style.format("{:.2f}"))
    
#     "***Les données utilisées ici sont des données fictives : elles ne sont pas au format de la suite de l'expérience.***"

