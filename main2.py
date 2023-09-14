#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:44:37 2023

Ici est encodé le protocole de la deuxième expérience.
INSÉRER LIEN PROTOCOLE.

Côté méthologique, on essaye de factoriser au mieux le code.
Je pars d'un copié collé de demo.py. (16/01/23)


@author: Corentin.Boidot
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import time
import warnings
from functools import reduce

from utils.d4data import (init_data_ev, init_data_tr, init_likelihood, 
                          init_simil, init_fi)

from utils.f4factorized import (display_match, display_score, display_shap,
                            display_likelihood, display_arg, 
                            feature_selection_shap, display_pastilles,
                            display_nearest,display_rule,make_df_pres,
                            )
from utils.p4pages import (page_pres,page_decision,page_interm,page_debrief,
                           decide2)


pd.set_option("display.max_columns",40)
pd.set_option("display.max_rows",30)

st.set_page_config(layout="wide")

# USER = 0 

#### initialisation des paramètres de session

path = "" #"../Exp1-Streamlit/" 
pathr = 'results2/'

N_tr = 25 # 15 # total disponible
N_ev = 50 # 16 # pour l'éval totale... # /!\ change le premier de l'eval 
N_cesure = 20 # 8 #sépare les deux "moments" de l'eval
transimatchs = [N_cesure//2,N_cesure,N_cesure+(N_ev-N_cesure)//2]
w_o_xp_tr = list(range(15,20)) # [11,13,14] # match du train sans interface

#%% ne pas executer
# il veut vivre

if 'abr' in st.session_state:
    N_ev = st.session_state.abr
    transimatchs = [N_cesure//2,N_cesure,N_cesure+(N_ev-N_cesure)//2]
    

if "user" not in st.session_state:
    st.session_state["user"] = int(input("N° de l'expérience : "))
USER = st.session_state["user"]

if "cur_page" not in st.session_state:
    print(USER)
    st.session_state["cur_page"] = 1 #8 # 13

if "cur_match" not in st.session_state:
    st.session_state["cur_match"] = 0


if "completed" not in st.session_state:
    st.session_state["completed"] = [False]*5
    
if "with_xp" not in st.session_state:
    st.session_state["with_xp"] = True

if "train" not in st.session_state:
    st.session_state['train'] = True

i = st.session_state.cur_match

if "reponses_num" not in st.session_state:
    st.session_state["reponses_num"] = pd.DataFrame(np.zeros((0,3)),
                            columns=["rep","time_print","time_click"])
    
if "click_list" not in st.session_state:
    st.session_state["click_list"] = []
    
if "reponses_form" not in st.session_state:
    st.session_state["reponses_form"] = pd.DataFrame(
                                            pd.Series(np.zeros((0))),
                                            columns=["rep"])

if "reponses_form2" not in st.session_state:
    st.session_state["reponses_form2"] = pd.DataFrame(
                                            # pd.Series(np.zeros((0))),
                                            columns=['confidence','difficulty',
                                                     'puA','puB','puC','puD'])
    
if "t_print" not in st.session_state:
    st.session_state["t_print"] = None
# utilisé pour obtenir l'instant où la page est affichée.
if "aide" not in st.session_state:
    st.session_state["aide"] = "neutre"
    
if "cur_clicks" not in st.session_state:
    st.session_state['cur_clicks'] = []
    
if 'NCALL' not in st.session_state:
    st.session_state['NCALL'] = 0
    
# if 'red' not in st.session_state:
#     st.session_state.red = 0
#     st.session_state.blue = 0

#%%init variables
                     
r1, r2, r3, r4, r5 = None, None, None, None, None    

# clickable = False

st.session_state.r = None


## Format réponses

options_ope = ["l’équipe bleue va certainement gagner",
            "l’équipe bleue va vraisemblablement gagner",
            "l’équipe bleue a un léger avantage sur l’équipe rouge",
            "je ne sais pas",
            "l’équipe rouge a un léger avantage sur l’équipe bleue",
            "l’équipe rouge va vraisemblablement gagner",
            "l’équipe rouge va certainement gagner"]
opt2 = [options_ope[1],'',options_ope[5]]

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

# "L’affichage {} répond à mes exigences en matière d’assistance IA." #explicable/transparente
# "L’affichage {} était facile à interpréter." #utiliser

q1 = pd.read_csv('questionnaire_ihm.csv',header=None)[0].to_list()
q2 = ["À quelle fréquence jouez-vous à League of Legends ?",
     "Quels rôles avez-vous déjà joué, jouez-vous usuellement ?",
     "Avez-vous déjà joué en équipe avec des amis ? (et si oui, combien)",
     "Depuis combien d'années jouez-vous à des MOBA ?",
     "Quel est votre dernier classement sur LoL ?",#"Quel est votre rang actuel dans LoL ?",
     "À quelle fréquence regardez-vous des matchs compétitifs ?",
     "Combien d'année vous êtes-vous intéressé à la méta de LoL ? (0 si vous l'ignorez)",
     "Avez-vous déjà publié du contenu grâce à LoL (vidéo, article) ? \
         Gagnez-vous de l'argent graĉe à LoL ?",
     "Quel est votre niveau de formation en informatique / sciences de l’ingénieur ?",
     "Avez-vous des connaissances en Intelligence Artificielle ?",
     
     "Vous êtes : ", "Quel âge avez-vous : ", "Y a-t-il d'autres points à \
         relever sur votre profil ou sur le déroulé de l'expérience ?"]

q1_idx = [4,7,5,4]+[8,4]
# indic = ["Tout au long de l’expérience, vous avez fait des pronostics en utilisant l’interface, qui présentait grâce aux boutons de gauche 4 *affichages IA* qui complétaient le pronostic de l’IA. Avant de recueillir une dernière fois vos ressentis sur les affichages séparément, nous allons vous demander de faire abstraction de ces affichages IA afin de noter l’interface en général (boutons “Suite”, curseur de décision, organisations des pages, présentation de la donnée, rappel des colonnes, interactions, déroulé général).",
#          "Quand vous appuyiez sur le bouton “Recommandation IA”, vous aviez accès à *l’interface IA* qui comprenait 4 *affichages*, ainsi qu’un “bouton radio” pour sélectionner l’affichage. Avant de noter individuellement ces affichages, nous allons vous demander de noter l'interface IA, c'est-à-dire les affichages considérés ensembles, avec la possibilité de naviguer entre eux.",
#          "Et maintenant, l'évaluation finale des affichages.",
#          "Concernant l'IA (niveau de l'interface IA)."]
indic = ["Il reste une dernière étape : remplir un formulaire. \n **Veuiller faire signe à l'expérimentateur.**",
         'Tout au long de l’expérience, vous avez fait des pronostics en utilisant le système de décision (cf image). \
             Quand l’IA était active, le bouton “Recommandation IA” vous donnait accès à l’interface IA, laquelle contenait 4 affichages. \n \
         Dans les pages qui suivent, nous allons vous demander d’évaluer d’abord \
             le système de décision (en faisant donc abstraction de l’interface IA), puis l’interface IA, et enfin les affichages séparément.',
             '','']

qo = ["Quelles sont vos attentes vis-à-vis de l’utilisation de l’IA dans un cadre d’application similaire ?",
      "Des remarques ?",
      "N’hésitez pas à noter ce qui vous a gêné dans la navigation.",
      "Comment avez-vous utilisé ces affichages ? Voyez-vous des points d’amélioration ?"
      ]

form_p = 14 # 1ere page de formulaire

#%% fonctions de navigation

def add_form(n,which=1):
    """
    n: int, nombre de lignes de formulaire à entrer
    """
    global r1 
    if which==1:
        f = 'reponses_form'
    elif which==2:
        
        f = 'reponses_form2'
    k = st.session_state[f].shape[0]
    # r = st.session_state['r']
    if n==1: #
        st.session_state[f].loc[k] = r1
    elif n==3: #
        st.session_state[f].loc[k] = r1
        st.session_state[f].loc[k+1] = r2
        st.session_state[f].loc[k+2] = r3
    elif n==5 and r5 is not None: #
        st.session_state[f].loc[k] = r1
        st.session_state[f].loc[k+1] = r2
        st.session_state[f].loc[k+2] = r3
        st.session_state[f].loc[k+3] = r4
        st.session_state[f].loc[k+4] = r5
    else:
        # if r1 is None:
        # r1 = st.session_state.r
        if which==2:
            rep = pd.DataFrame(np.array([r1]),columns=['confidence','difficulty',
                                                 'puA','puB','puC','puD'])
            st.session_state[f] = \
                    pd.concat([st.session_state[f],rep])
        elif which==1:
            rep = pd.DataFrame(np.array(r1),columns=['rep'])
            st.session_state[f] = \
                    pd.concat([st.session_state[f],rep],)
        # print('form')
        # print(st.session_state[f])


def save_num(phase):
    df = st.session_state.reponses_num
    df.to_csv(pathr+"reponses_matchs_" + phase 
              + '_' +  str(USER) + ".csv", index=False)
    st.session_state['reponses_num'] = pd.DataFrame(np.zeros((0,3)),
                            columns=["rep","time_print","time_click"])
    with open(pathr+"clicks_"+phase+'_'+str(USER)+'.pkl','wb') as f:
        pickle.dump(st.session_state['cur_clicks'],f)
    st.session_state['cur_clicks'] = []
    st.session_state.reponses_form2.reset_index(drop=True).to_csv(
        pathr+"reponses_form2_" + str(st.session_state.user) + ".csv")
    print("Phase {} saved !".format(phase))

def abreger(n=40):
    if n is not None:
        st.session_state['abr'] = n 
    else:
        print("Problem during abreger. Ignoring.")

def nextpage(out=False): #back=0 #ignore_c=False

    # global NCALL 
    # print(st.session_state.NCALL)
    st.session_state.NCALL += 1
    if st.session_state.NCALL > 1:
    # if not (clickable or ignore_c) :
        time.sleep(.2) # print("Ne cliquez pas si vite !")
        if st.session_state.NCALL >=3:
            raise Exception("Arrête de clicker comme un fils de p***")
    else:
        global r1
        # print(r1)
        clickable = False
        sscp = st.session_state.cur_page
        # print('agnagna')
        # print("current page : {}".format(sscp))
            # if sscp<8:
            #     st.session_state.cur_page += 1
        if sscp==8: # réponse question
        # sscp==6: # début des cas spéciaux, sinon +1
            line = [r1, st.session_state.t_print, time.time()]
            st.session_state['reponses_num'].loc[st.session_state.cur_match] = line
            st.session_state['cur_clicks'].append((time.time(),st.session_state.aide))
            st.session_state.cur_page += 1
            # input("vas-y")
            st.session_state['t_print'] = None
            st.session_state.cur_match += 1
            
        elif sscp==9: # intercalaire
            add_form(7,which=2)
            if st.session_state.train:
                if (st.session_state.cur_match)%5==0:
                    st.session_state.cur_page += 1 # again
                else:
                    st.session_state.cur_page -= 1
                st.session_state.with_xp = not st.session_state.cur_match in w_o_xp_tr
            else:
                if st.session_state.cur_match in transimatchs:
                    st.session_state.cur_page = 13
                elif st.session_state.cur_match == N_ev:
                    st.session_state.cur_page = 11
                else:
                    st.session_state.cur_page -= 1
        elif sscp==10: # debrief
            if st.session_state.cur_match%N_tr==0:
                st.session_state.cur_page += 1
            # if st.session_state.cur_match<29:#back==1:
            #     st.session_state.cur_page = 8 #
            else:
                st.session_state.cur_page = 8
                # st.session_state.cur_match += 1
        elif sscp == 11: # questionnaire interphase
            add_form(3)
            if st.session_state.train:
                save_num('tr')
                st.session_state.cur_page += 1
                if int(st.session_state.user)%2 == 0:
                    st.session_state["with_xp"] = False 
                else:
                    st.session_state["with_xp"] = True
            else:
                save_num('ev')
                st.session_state.cur_page = 14
        elif sscp==12:
            if st.session_state.train==True:
                st.session_state.train = False
                st.session_state.cur_match = 0
                st.session_state.part = 2
                st.session_state.with_xp
                st.session_state.cur_page = 8
            else: # jamais utilisé ici
                st.session_state.cur_page += 1
        elif sscp==13:
            st.session_state.with_xp = not st.session_state.with_xp
            if r2 is not None:
                abreger(r2)
            st.session_state.cur_page = 8
        else: 
            # pages finales de formulaire
            if sscp >= form_p: 
                n = form_p-sscp
                add_form(q1_idx[n])
            if sscp == 19: # mtnt le grand FINAL !
                with open(pathr+"times_tr_" + str(USER) + ".pkl",'wb') as f:
                    pickle.dump(st.session_state.click_list,f)
                # on reindexe par les questions de formulaire. (ou pas)
                # st.session_state['reponses_form'] = \
                #     st.session_state.reponses_form.reset_index(drop=True).rename(
                #         index=lambda x:  (list(range(12))+q1+q2)[x])
                st.session_state.reponses_form.to_csv(
                    pathr+"reponses_formulaire_" + str(st.session_state.user) + ".csv")
                # st.session_state.reponses_form.to_csv(
                #     pathr+"reponses_form2_" + str(st.session_state.user) + ".csv")
            # Default behavior
            st.session_state.cur_page += 1
            
        st.session_state.click_list.append(time.time())        
        st.session_state['t_print'] = None
        st.session_state["completed"] = [False]*5
        st.session_state['aide'] = 'neutre'
        # st.session_state.red = 0
        # st.session_state.blue = 0
        # st.session_state.NCALL = 0 # je le mets en bas de page
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        if out:
            st.experimental_rerun()
        time.sleep(.1)

#%% Chargement des données %%#

np.set_printoptions(suppress=True)

df_pres = make_df_pres()

df_tr, labels = st.cache(init_data_tr)(USER,N_tr)
df_ev = st.cache(init_data_ev)()
xpo_tr, xpo_ev, pred_tr, pred_ev = init_fi(N_tr,df_tr)
likeli_tr,likeli_ev = init_likelihood()
# near_tr,near_ev = init_simil() #, ann2_tr,ann2_ev  alter=True
near1_tr,near1_ev, ann1_tr, ann1_ev = init_simil(1,alter=True)
near1_tr = near1_tr.reindex(index=df_tr.drop('moy').index)
ann1_tr = ann1_tr.reindex(index=df_tr.drop('moy').index)
lime_tr, lime_ev, _, _ = init_fi(N_tr,df_tr,"lime")


if st.session_state.train:
    params = {
        "df": df_tr,
        "pred": pred_tr,
        "sv": xpo_tr,
        "like": likeli_tr,
        # "near": near_tr,
        "near1": near1_tr,
        "lime": lime_tr,
        "ann1": ann1_ev,
        "N": N_tr,
        }
else:
    params = {
        "df": df_ev,
        "pred": pred_ev,
        "sv": xpo_ev,
        "like": likeli_ev,
        # "near": near_ev,
        "near1": near1_ev,
        "lime": lime_ev,
        "ann1": ann1_ev,
        "N": N_ev,
        }
params["df_pres"] = df_pres
params["labels"] = labels
    
N = params['N']
sv = params['sv']
near1 = params['near1']
# near = params['near']
lime = params['lime']
ann1 = params['ann1']
pred = params['pred']
df = params['df']


navig = nextpage
# oc = record
# xp_options = ["confiance","texte","math","voisin"]

##############################################################################
#%%             ICI COMMENCE L'INTERACTION AVEC L'UTILISATEUR                #
##############################################################################

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
    "L'expérience va se dérouler en 2 phases, et vise à mesurer \
        vos performances prédictives sur nos données de match. Vous aurez une IA dotée de différentes  \
            interfaces pour vous aider. La première phase est un entraînement \
                pour pouvoir vous approprier la tâche : essayez d'identifier \
                    l'utilisation de l'interface qui vous conviendra le mieux, avant la deuxième phase qui \
                        constitue l'évaluation réelle (vous n'aurez plus de retour direct sur vos performances)."
    "Nous évaluons deux critères (sans donner de priorité) : la précision des réponses et la vitesse de réponse."
    # /!\ définir plus d'enjeux ? /!\
    "Cliquez sur START pour continuer."
    
    st.button("START>>>",key='nekst',on_click=nextpage) 
    # i = st.slider("aller à la page")  ## pour débugger
    # st.button("GO!",on_click=cheatcode,args=[i])  ## débugger
    
elif st.session_state.cur_page == 2:
    
    st.title("Présentation des données statistiques")
    
    left_column, right_column = st.columns(2)
    with left_column:
        "Durant l'expérience, vous aller devoir traiter des données au format \
            ci-contre. Vous y trouverez les relevés de l'équipe bleue (colonne 'blue'), \
                de l'équipe rouge (colonne 'red')."
        "La colonne 'mean' contient la moyenne statistique de la valeur \
            mesurée, à partir de l'ensemble des matchs à disposition."
        "À gauche, vous avez une description des lignes que vous pouvez \
            masquer si besoin."
    display_match(1,df,col=right_column)
        
    st.sidebar.table(df_pres)
    # st.session_state["train"] = True
    
    left_column.button("Suite>>>",key='nekst',on_click=nextpage) 

elif st.session_state.cur_page == 3:
    
    st.title("Instructions pour l'entraînement")
    # left_column, right_column = st.columns(2)
    bonus, left_column, right_column = st.columns([1,3,3]) # ou 3 ?
    with left_column:
        "L'assistance IA vous sera proposée grâce au bouton à gauche."
        "L'IA produit un score de prédiction continu sous forme de \
            probabilité de victoire, qui est sa présentation par défaut \
                            (ci-contre à droite), qui vous sera donc \
                présenté en premier quand vous activerez l'IA."
        "Vous aurez aussi accès à trois explications différentes de cette \
            recommandation IA, que nous allons vous présenter dans les pages suivantes."
        "Vous prendrez vos décisions à l'aide d'un bouton en bas de page. \
            Le bouton pour passer à la page suivante se trouve à gauche."
        # "Vous aurez accès à cette prédiction \
        #         ainsi qu'à une interface d'analyse."
        # "Cette interface montre les valeurs les plus importantes dans la \
        #     décision du modèle, et dans quel sens chacune a contribué."
        # "Ces contribution sont représenté ci-dessous par des barres \
        #     de longueur proportionnelle à l'importance."
        # "La barre est rouge si la variable a contribué au bon score de \
        #     l'équipe rouge (et respectivement bleue pour l'équipe bleue)."
        
        # st.radio('Quel est votre pronostic ?',opt2)
    # display_shap(1,lime_tr,col=right_column,plot='flot')
    decide2(title="Exemple de décision",col=left_column)
    display_score(i, pred,col=right_column,title=True)  
    with bonus:
        if st.button("Recommandation IA"):
            "vous pourrez bientôt tester les autres possibilités d'utilisation \
                de l'interface IA. Veuillez cliquer sur Suite."
                # "Celle-ci produit un score de prédiction continu, sous forme de \
                #     probabilité de victoire. Vous aurez accès à cette prédiction \
                #         ainsi qu'à une interface d'analyse"
                # "Cette interface montre les valeurs les plus importantes dans la \
                #     décision du modèle, et dans quel sens chacune a contribué"
                # "Ces contribution sont représenté ci-dessous par des barres \
                #     de longueur proportionnelle à l'importance."
                # "La barre est rouge si la variable a contribué au bon score de \
                #     l'équipe rouge (et respectivement bleue pour l'équipe bleue)."
                # display_shap(1,col=right_column)
    bonus.button("Suite>>>",key='nekst',on_click=navig)    
    st.sidebar.table(df_pres)

elif st.session_state.cur_page == 4:
    
    st.title("Règle de décision IA")
    txt = "L'interface vous propose une description de la décision de l'IA \
        par une règle simplifiée."
    page_pres(txt,1,1,navig,**params)
    st.sidebar.table(df_pres)
    
elif st.session_state.cur_page == 5:
        
   st.title("Importance de variables IA")
   txt = "L'interface vous propose une quantification du poids des variables \
       dans son pronostic : une barre bleue fait pencher les probabilités en \
           faveur de l'équipe bleue, une barre rouge en faveur de l'équipe rouge. \
               L'axe horizontale correspond approximativement à la probabilité \
                   de victoire de l'équipe rouge selon l'IA (une valeur 0.1 correspond \
                        ainsi à 90% de chance de victoire pour les bleus)."
   page_pres(txt,2,1,navig,**params)
   st.sidebar.table(df_pres)

elif st.session_state.cur_page == 6:
        
    st.title("Plus proche voisin IA")
    txt = "L'interface vous propose un exemple passé, similaire aux yeux de l'IA.\
        Attention ! L'IA peut recommander une issue différente de celle de ce match. \
        Le texte en dessous présente la prédiction de l'IA sur ce cas, ainsi que le \
            résultat du match (quelle équipe avait effectivement gagné). \
                En moyenne, les trois plus proches voisins d'un match sont \
                    gagnés par la même équipe : l'IA indique combien de matchs \
                        proches elles doit effectivement chercher pour observer une issue différente. "
    txt += "Ces 'issues' se basent des victoires réelles, et non sur les prédictions de l'IA."
    page_pres(txt,3,i=1,navig=navig,**params)
    st.sidebar.table(df_pres)

elif st.session_state.cur_page == 7:
    
    "Vous allez commencez la phase d'entraînement."
    "Vous aurez un accès constant à l'IA, mais vous pouvez aussi tenter de \
        faire vos pronostics en vous en passant. Après chaque match, n'hésitez \
        pas à prendre des notes sur votre utilisation de l'interface IA."
    "Tous les 5 matchs, vous pourez revenir en arrière pour vérifier vos \
        réponses ainsi que les suggestions de l'IA."
    "À partir du 10e match, on vous affichera également votre temps de décision. \
        Il faut alors prendre ses décisions aussi vite que possible."
    "N'hésitez pas pour cela à adapter votre choix d'interface au match que vous \
        observez."
    
    st.button("Suite>>>",key='nekst',on_click=navig)    
        
elif st.session_state.cur_page == 8:
    
    i = st.session_state.cur_match
    a = 0 if st.session_state.train else -1
    tr_case = st.session_state.train and i in w_o_xp_tr
    if not st.session_state.with_xp or tr_case:
        r1 = page_decision(i+1,True,navig,ajust=a,disable=True,**params)
    else:
        r1 = page_decision(i+1,True,navig,ajust=a,**params)
    # st.button("NEXT>>>",key='nekst',on_click=navig)    

elif st.session_state.cur_page == 9:
    
    # INTERMEDIAIRE
    
    if st.session_state.train:
        i_ = i # i = cur_match... 
        extrait = st.session_state.reponses_num.loc[:,'rep']
        # exprimes = extrait.apply(lambda x: False if x=="je ne sais pas" else True)
        numerateur = extrait.sum() # exprime
        choix = extrait.apply(lambda x: interp_ope[x]) # [exprimes]
        # # labels.loc[i+5:i+9][exprimes]
        # dividende = (choix.values == labels.values[:i]).sum()
        dividende = np.equal(choix.values,labels.values[:i]).sum()
        # ajust # enlevés
        score = (dividende,extrait.shape[0])
    
    # page_interm(navig,score=score,ajust=-1,**params)
    if not st.session_state.train:
        r1 = page_interm(navig,**params)
    elif st.session_state.cur_match <=5: #10
        r1 = page_interm(navig,score=score,**params)
    else : # st.session_state.cur_match <=20
        dt = (st.session_state.reponses_num['time_click'] - \
              st.session_state.reponses_num['time_print']).loc[i-1]
        r1 = page_interm(navig,score=score,dt=dt,**params)
    # elif st.sessions_state.cur_match <=20:
    #     page_interm(navig,**params)

elif st.session_state.cur_page == 10:
    
    # DEBRIEF
    
    if st.session_state.cur_match >= 29:
        bonus_txt = "Fin de l'entraînement."
    elif st.session_state.cur_match == 10:
        bonus_txt = "Pour la suite, faites attention à votre temps de décision ! \
                    Essayez d'adapter votre choix d'interface pour atteindre les \
                        meilleures décisions le plus vite possible."
    elif st.session_state.cur_match == 15:
        bonus_txt = "Vous n'aurez pas accès à l'interface durant les prochains matchs."
    else:
        bonus_txt = "Vous pouvez poursuivre l'entraînement."
        
    a = -0 if st.session_state.train else -1
    if st.session_state.cur_match <=5: #0
        page_debrief(navig,ajust=a,bonus_txt=bonus_txt,**params)#-1
    else : # st.session_state.cur_match <=20
        dt = (st.session_state.reponses_num['time_click'] - \
              st.session_state.reponses_num['time_print']).mean() #.iloc[:-10]
        page_debrief(navig,ajust=a,dt=dt,bonus_txt=bonus_txt,**params)#-1

elif st.session_state.cur_page == 11:
    
    if not st.session_state.train:
        st.title("Fin de la phase d'évaluation")
    else:
        st.title("Fin de la phase d'entraînement")
    
    "Nous allons recueillir vos ressentis."
    
    r1 = st.select_slider("Pensez-vous avoir progressé depuis le début de l’expérience ?",
                          nondutout,"indécis")
    r2 = st.select_slider("Étiez-vous très concentré durant l’expérience ?",
                          nondutout,"indécis")
    r3 = st.select_slider("Êtes-vous fatigué ?",
                          nondutout,"indécis")
    
    # if not st.session_state.train:
    #     "Il reste une dernière étape : remplir un formulaire."
    #     st.markdown("**Veuiller faire signe à l'expérimentateur.**")
    ### Pas assez visible
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)

elif st.session_state.cur_page == 12:

    st.title("La phase d'évaluation va commencer")
    
    "Vous allez passer à l'évaluation réelle de votre capacité de pronostic.\n"
    "Objectif : prédire l'équipe gagnante d'un match de League of Legends \
        à partir de données statistiques à 10 minutes du match."
    st.write("Vous aurez {} matchs à évaluer : **vous pouvez prendre une pause entre \
        deux pronostics** au besoin, mais une fois que l'écran de match s'affiche, \
            tentez de **répondre au plus vite**.".format(N_ev))
    "Attention : vous n'aurez pas toujours accès à l'IA." # TO DO
    if st.session_state.with_xp:
        "En l'occurence, vous aurez accès à l'IA pour les premiers matchs."
    else:
        "En l'occurence, vous commencez sans IA, avec les données seules."
    # "Évitez autant que possible de passer plus d'une minute sur un match."
    
    st.button("START>>>",key='nekst',on_click=nextpage)
    
elif st.session_state.cur_page == 13:
    
    st.title("Changement de l'interface")
    if st.session_state.cur_match == N_cesure:
        dt = (st.session_state.reponses_num['time_click'] - \
              st.session_state.reponses_num['time_print']).mean() #.iloc[:-10]
        txt = "Votre temps de décision moyen est de {0:.2f} secondes en \
                 l'évaluation, pour l'instant.".format(dt)
        st.markdown("**Veuiller faire signe à l'expérimentateur.**")
        # r2 = st.radio(txt,['0','1'],key='long') # ,on_change=abreger
        r2 = st.number_input("À REMPLIR PAR L'EXPERIMENTATEUR",24,50,40)
        st.write(r2)
    if st.session_state.with_xp:
        "Vous n'aurez plus accès à l'IA pour les prochains matchs."
    else:
        "Vous aurez accès à l'IA pour les prochains matchs."
    st.button("Suite>>>",key='nekst',on_click=nextpage)
    st.sidebar.table(df_pres)

elif 18 > st.session_state.cur_page >= form_p:
    
    n = st.session_state.cur_page - form_p
    # st.title("FROMULAIRE")
    txt = qo[n]
    if st.session_state.cur_page == 15:
        l,r = st.columns([3,4])
        l.markdown(indic[n])
        r.image('nomenclature_jolie.png')
    else:
        if len(indic[n])>0:
            st.markdown(indic[n])
    # else:
    #     st.markdown(indic[n])
    N = q1_idx[n]
    r1 = [None]*N
    start = sum([q1_idx[x] for x in range(n)])
    # st.write("n {} N {} start {}".format(n,N,start))
    for i in range(N):
        # st.write(i)
        r1[i] = st.select_slider(str(q1[start+i]), pasdaccord,'neutre')
    st.text_input(qo[n],disabled=True)
    st.button("Suite>>>",key='nekst',on_click=nextpage)

# elif st.session_state.cur_page == 15:
#     txt = ''
#     st.markdown(txt)
#     st.text_input('',disabled=True)
#     st.button("Suite>>>",key='nekst',on_click=nextpage)

# elif st.session_state.cur_page == 16:
    
#     st.text_input("N’hésitez pas à indiquer les points d’amélioration que vous verriez sur certains de ces affichages.",disabled=True)
#     st.button("Suite>>>",key='nekst',on_click=nextpage)
    
# elif st.session_state.cur_page == 17:
    
#     st.text_input("Quelles sont vos attentes vis-à-vis de l’utilisation de l’IA dans un cadre d’application similaire ?",disabled=True)
#     st.button("Suite>>>",key='nekst',on_click=nextpage)
    
elif st.session_state.cur_page == 18:
    
    r1 = [0]*8
    
    r1[0] = st.selectbox(q2[0], temp_jeu)
    r1[1] = st.text_input(q2[1])
    r1[2] = st.selectbox(q2[2],["juste en solo",1,2,4])
    r1[3] = st.number_input(q2[3], max_value=22,format="%d")
    r1[4] = st.selectbox(q2[4], rangs)
    r1[5] = st.selectbox(q2[5],temp_visio)
    r1[6] = st.number_input(q2[6], max_value=15,format="%d")
    r1[7] = st.text_input(q2[7],disabled=True)
    st.button("Suite>>>",key='nekst',on_click=nextpage)

elif st.session_state.cur_page == 19:
    
    r1 = [0]*4
    r1[0] = st.selectbox(q2[8], ["Aucun", "Bac spé math/info", "Bac+2", "Bac+3 (équivalent licence)", 
                           "Bac+4", "Bac+5 (équivalent master)", "Bac+6", "supérieur à Bac+6"])
    r1[1] = st.selectbox(q2[9], ["Aucune", "en amateur (par la vulgarisation)",
                          "passionné d’informatique et de statistique (connaissances techniques)", 
                    "jeune pro (M2 spécialisé ou doctorant)", "confirmé (plus de trois ans \
                        de travail dans le domaine du Machine Learning"])
    r1[2] = st.selectbox("Vous êtes : ", ["un homme","une femme","préfère ne pas répondre"])
    r1[3] = st.text_input("Quel âge avez-vous : ")
    r0 = st.text_input("Y a-t-il d'autres points à relever sur votre profil ou \
                       sur le déroulé de l'expérience ?",disabled=True)
    st.button("Suite>>>",key='nekst',on_click=nextpage)

elif st.session_state.cur_page == 20:
    st.title("Merci.")
    "Félicitation, vous avez terminé l'expérience ! Au nom de toute l'équipe de \
        recherche, je vous remercie pour votre contribution."
    "Et après l'effort le réconfort ;)"
    

if st.session_state.t_print is None: # permet le calcul du temps d'affichage.
    st.session_state['t_print'] = time.time()
    # print("Time updated")
    if st.session_state.cur_page == 8: # decision
        st.session_state['cur_clicks'].append((time.time(),"start")) # st.session_state.aide
    
st.write("page : "+str(st.session_state.cur_page))
st.session_state['NCALL'] = 0
clickable = True

# st.write(st.session_state)

# st.session_state.cur_match = st.number_input("GO TO : ",st.session_state.cur_match)
# très mauvaise idée