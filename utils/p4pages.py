#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 08:59:37 2023

@author: Corentin.Boidot
"""
import time

from .f4factorized import (display_match, display_score, display_shap,
                            display_likelihood, display_arg, 
                            feature_selection_shap, display_pastilles,
                            display_nearest,display_rule,make_df_pres,
                            options_ope,nondutout,pasdaccord,
                            )
import streamlit as st

TEXT_DISABLE = True

xp_options = ["confiance","texte","math","voisin"]
xp_options = ["A","B","C","D"]

interm = ["J'étais confiant en ma décision.",
          'Le match était difficile à analyser.'] + \
          ["L'interface '"+x+"' m'a été utile." for x in xp_options] + \
         ["Autres remarques : "]
         #  "Le score de confiance m'a été utile."] + \
         # ["L'explication '"+x+"' m'a été utile." for x in xp_options[1:]] + \

interp_ope = {"l’équipe bleue va certainement gagner":0,
            "l’équipe bleue va vraisemblablement gagner":0,
            "l’équipe bleue a un léger avantage sur l’équipe rouge":0,
            "l’équipe rouge a un léger avantage sur l’équipe bleue":1,
            "l’équipe rouge va vraisemblablement gagner":1,
            "l’équipe rouge va certainement gagner":1}
opt2 = ['l’équipe bleue va vraisemblablement gagner', '',
        'l’équipe rouge va vraisemblablement gagner']

available_xp = ['confiance',"bar-SHAP","texte","probas","pastilles","simil2",
                "simil1",'rule','lime-line','lime-flot',"lime-bar",'lime-pastille']

    
# selection = ['confiance','rule']
# aliases = [['A']['B']['C']['D']]
dic_xp = {'A':"confiance",#selection[0],
          'B':'rule', 'texte':'rule',
          'C':'lime-flot', 'math':'lime-flot',#'bar-SHAP'
          'D':'simil1', 'voisin':'simil1'}
    
def core_expe2(i,mode,ajust,col,**kwargs):
    N = kwargs['N']
    sv = kwargs['sv']
    near1 = kwargs['near1']
    lime = kwargs['lime']
    ann1 = kwargs['ann1']
    pred = kwargs['pred']
    df = kwargs['df']
    if mode not in available_xp:
        mode = dic_xp[mode]
    hide = (mode!='confiance')
    display_score((i+ajust)%N,pred,col=col,wording=2,hide_score=hide,title=True)
    if mode == "bar-SHAP":
        display_shap((i+ajust)%N,sv,col=col)
    elif mode == "pastilles":
        display_pastilles((i+ajust)%N,sv,n=4)
    elif mode == "simil1":
        display_nearest((i+ajust)%N,near1,ann=ann1,col=col)#,df.loc['moy']
    elif mode == 'rule':
        display_rule((i+ajust)%N,df,pred,col=col,w=2)
    elif mode == "lime-flot":
        display_shap((i+ajust)%N,lime,col=col,plot='flot')
        
def decide_old(**kwargs):
    
    pass
    # return r1,True

def decide2(col,title='Quel est votre pronostic ?',opt=opt2,default=''):
    r1 = col.select_slider(title,opt,default,key="slide")
    dis = (r1==default)
    return r1, dis

def record(aide=None):
    if aide is None:
        st.session_state['cur_clicks'].append((time.time(),
                                           st.session_state.aide))
    elif aide in ['neutre']+xp_options:
        st.session_state['cur_clicks'].append((time.time(),aide))
    else:
        st.session_state['cur_clicks'].append((time.time(),
                                           st.session_state[aide]))
    
def show_xpo():
    st.session_state['aide'] = 'confiance'
    record('neutre')

#%% pages

def page_pres(txt,option:int,i:int,navig,rec=None,**kwargs):
    # option 
    # txt
    N = kwargs['N']
    pred = kwargs['pred']
    df = kwargs['df']
    
    bonus, left_column, right_column = st.columns([1,3,3]) # ou 3 ?
    
    mode = st.session_state.aide
    left_column.write(txt)

    with bonus:
        pass
    ajust = 0
    if True:
        mode = bonus.radio("Mode de l'interface", xp_options,index=option,
                           key='pasaide',on_change=rec,disabled=True)
        mode = xp_options[option]
        with right_column:
            core_expe2(i, mode, ajust, right_column, **kwargs)
    bonus.button("Suite>>>",on_click=navig)
    
    # left_column.title("Match n°0")
    display_match((i+ajust-1)%N,df,col=left_column)
    title = "ce slider vous permettra de prendre une décision"
    _,_ = decide2(left_column,title=title)

def page_decision(i,with_xp,navig,ajust=0,
                  turn_on=show_xpo,rec=record,disable=False,
                  **kwargs):
    
    N = kwargs['N']
    df = kwargs['df']
    df_pres = kwargs['df_pres']
    
    bonus, left_column, right_column = st.columns([1,3,3]) # ou 3 ?
    # i = left_column.selectbox("Choisissez le match à afficher",
    #                  list(range(1,N_tr+1)))
    # i = st.session_state.cur_match # -4 # /!\ que faire ?
    mode = st.session_state.aide
    # st.write(mode)

    with bonus:
        if mode == "neutre":# and st.session_state.with_xp: # 
            st.button("Recommandation IA",on_click=turn_on,disabled=disable)
            mode = st.session_state.aide
    # ajust = 0
        # winner = 'rouge' if labels.iloc[(i+ajust-1)%N_tr] else 'bleue'
        # st.title("Équipe gagnante : " + winner)
        # predi = st.session_state.reponses_num.loc[i+4,'rep']
        # st.title("Votre pronostic : " + predi)
        # exp = st.expander('explication IA',expanded=True)
    if mode != "neutre":
        mode = bonus.radio("Mode de l'interface",xp_options,key='affi',on_change=rec)
        st.session_state.aide = mode
        with right_column:
            col = right_column
            core_expe2(i, mode, ajust, right_column, **kwargs)
            # display_score((i+ajust)%N,pred,col=col,wording=2)
            # mode = st.radio("Mode de l'interface", ["confiance","voisin",
            #                          "math","texte"],   )
            # st.session_state.aide = mode
    
    left_column.title("Match n°"+str(i))
    display_match((i+ajust-1)%N,df,col=left_column)
    # vus = st.session_state.completed
    # vus[(i-1)%5] = True
    r1, dis = decide2(left_column)
    
    if not dis:
        bonus.button("Valider",on_click=navig)
    st.sidebar.table(df_pres)
    
    return r1

def page_interm(navig,score=None,dt=None,**kwargs): #txt#option:int,i:int
    
    df_pres = kwargs['df_pres']
    if score is not None:
        st.title("{} bonnes réponses sur {}".format(score[0],score[1]))
    else:
        st.title("Vos retours sont précieux !")
    if dt is not None:
        st.markdown("### Votre temps de décision : {0:.2f} secondes.".format(dt))
    # Print "Bonne réponse !" , éventuellement le temps
    
    # with st.form('interm'):
    r1 = [0]*6 #7
    st.session_state.r = r1
    # n_q = 2
    # if st.session_state.with_xp:
    #     n_q = 6
    # for i in range(n_q):
    #     r1[i] = st.select_slider(interm[i],pasdaccord,'neutre',key=str(i))
    for i in range(2):
        r1[i] = st.select_slider(interm[i],pasdaccord,'neutre',key=str(i))
    if st.session_state.with_xp:
        for i in range(2,6):
            r1[i] = st.radio(interm[i],['non','oui'],key=str(i)) #horizontal=
            # multiselect
    st.text_input(interm[6],key='haha',disabled=TEXT_DISABLE) #r1[6] = 
    
    st.button("Reprendre>>",on_click=navig) #:  # form_submit_
        # r1[6] = st.session_state.fuck # tant pis
        # st.session_state.r = r1
        # st.session_state['cache'] = st.session_state.fuck
        # if r1[6] != st.session_state['cache']:
        #     print('LOL')
        #     print("Here is r1 : {}".format(r1))
        #     print(st.session_state.fuck)
        # navig(out=True) # ,on_click=navig

    # if st.button('non'):
    #     st.experimental_rerun()
    #     st.
    #     st.write('oui')
    # st.write(r1)
    st.sidebar.table(df_pres)
    return r1


def page_debrief(navig,ajust=0,dt=None,bonus_txt='',**kwargs):
    
    N_tr = kwargs['N']
    labels = kwargs['labels']
    df_pres = kwargs['df_pres']
    df = kwargs['df']
    # left_column, right_column = st.columns(2) # ou 3 ?
    bonus, left_column, right_column = st.columns([1,3,3])
    # if st.session_state.part==2:
    #     ajust = 5 + st.session_state.end_tr1 # hé, il faut vivre avec son temps
    # else:
    if dt is None:
        left_column.title("N'hésitez pas à retourner voir vos décisions précédentes !")
    else:
        left_column.title("Votre temps de décision moyen : {0:.2f} secondes".format(dt))
    i_ = st.session_state.cur_match -5 +ajust
    with right_column:
        # 'Votre score : ' + str(dividende)+" / "+str(numerateur)
        # /!\ je réutilise i à une autre fin
        i = st.selectbox("Choisissez le match à afficher",
                         [i_+1,i_+2,i_+3,i_+4,i_+5])
        # vus = st.session_state.completed
        # vus[(i-1)%5] = True
        
        predi = st.session_state.reponses_num.loc[i-1,'rep']
        st.title("Votre pronostic : " + predi)
    # if st.session_state.with_xp:
    mode = bonus.radio("Mode de l'interface",xp_options,key='pasaide')#,on_change=record
    st.session_state.aide = mode
    # display_score((i+ajust)%N_tr,col=right_column,wording=2)
    # display_shap((i+ajust)%N_tr,col=right_column)
    core_expe2(i, mode, ajust, right_column, **kwargs)
    bonus.write(bonus_txt)
    bonus.button("Suite>>>",on_click=navig)
    
    display_match((i+ajust-1)%N_tr,df,col=left_column)
    winner = 'rouge' if labels.iloc[(i+ajust-1)%N_tr] else 'bleue'
    left_column.title("Équipe gagnante : " + winner)
    # if not isok:
    #     left_column.write("Vous devez réussir au moins 75% de vos prédictions, \
    #                       avec moins de trois 'je ne sais pas'.")
    
    st.sidebar.table(df_pres)
    

    
#%% ctrl+X crtl+V de main2

# comme un bourrin

# le propre là-bas doit être code

# el
if False:# st.session_state.cur_page == 56:
        
    left_column, right_column, bonus = st.columns([3,3,1]) # ou 3 ?
    
    # i = left_column.selectbox("Choisissez le match à afficher",
    #                  list(range(1,N_tr+1)))
    i=5
    mode = st.session_state.aide
    # st.write(mode)

    with bonus:
        
        if mode == "neutre":
            st.button("Recommandation IA",key='pasaide',on_click=show_xpo)
            # mode = st.session_state.aide
    ajust = 0
        # winner = 'rouge' if labels.iloc[(i+ajust-1)%N_tr] else 'bleue'
        # st.title("Équipe gagnante : " + winner)
        # predi = st.session_state.reponses_num.loc[i+4,'rep']
        # st.title("Votre pronostic : " + predi)
        # exp = st.expander('explication IA',expanded=True)   
    if mode != "neutre":
        mode = bonus.radio("Mode de l'interface", ["confiance","voisin",
                                 "math","texte"],on_click=record)
        # st.session_state.aide = mode
        with right_column:
            col = right_column
            title = "ce slider ne prends pas de décision"
            st.select_slider(title,options_ope,"je ne sais pas",key="slide")
            if mode == "bar-SHAP":
                display_shap((i+ajust)%N_tr,sv,col=col)
            elif mode == "pastilles":
                display_pastilles((i+ajust)%N_tr,sv,n=4)
            elif mode == "voisin":
                display_nearest((i+ajust)%N_tr,near,df.loc['moy'],col=col)
            elif mode == 'texte':
                display_rule((i+ajust)%N_tr,df,pred,col=col)
            elif mode == "math":
                display_shap((i+ajust)%N_tr,lime,col=col,plot='flot')
                
            display_score((i+ajust)%N_tr,pred,col=col,wording=2)
            # mode = st.radio("Mode de l'interface", ["confiance","voisin",
            #                          "math","texte"],   )
            # st.session_state.aide = mode
        # else:
    bonus.button("Suite>>>",on_click=nextpage)
    
    left_column.title("Match n° osef")
    display_match((i+ajust-1)%N_tr,df,col=left_column)
    
    st.sidebar.table(df_pres)
    
# elif st.session_state.cur_page == 777:
        
#     bonus, left_column, right_column = st.columns([1,3,3]) # ou 3 ?
    
#     # i = left_column.selectbox("Choisissez le match à afficher",
#     #                  list(range(1,N_tr+1)))
#     i=5
#     mode = st.session_state.aide
#     # st.write(mode)

#     with bonus:
        
#         if mode == "neutre":
#             st.button("Recommandation IA",on_click=show_xpo,)
#             mode = st.session_state.aide
#     ajust = 0
#         # winner = 'rouge' if labels.iloc[(i+ajust-1)%N_tr] else 'bleue'
#         # st.title("Équipe gagnante : " + winner)
#         # predi = st.session_state.reponses_num.loc[i+4,'rep']
#         # st.title("Votre pronostic : " + predi)
#         # exp = st.expander('explication IA',expanded=True)   
#     if mode != "neutre":
#         mode = bonus.radio("Mode de l'interface", ["confiance","voisin",
#                                  "math","texte"])
#         # st.session_state.aide = mode
#         with right_column:
#             col = right_column
#             if mode == "bar-SHAP":
#                 display_shap((i+ajust)%N_tr,sv,col=col)
#             elif mode == "pastilles":
#                 display_pastilles((i+ajust)%N_tr,sv,n=4)
#             elif mode == "voisin":
#                 display_nearest((i+ajust)%N_tr,near,df.loc['moy'],col=col)
#             elif mode == 'texte':
#                 display_rule((i+ajust)%N_tr,df,pred,col=col)
#             elif mode == "math":
#                 display_shap((i+ajust)%N_tr,lime,col=col,plot='flot')
                
#             display_score((i+ajust)%N_tr,pred,col=col,wording=2)
#             # mode = st.radio("Mode de l'interface", ["confiance","voisin",
#             #                          "math","texte"],   )
#             # st.session_state.aide = mode
#         # else:
#     bonus.button("Suite>>>",on_click=nextpage)
    
#     left_column.title("Match n° osef")
#     display_match((i+ajust-1)%N_tr,df,col=left_column)
#     title = "ce slider ne prends pas de décision"
#     left_column.select_slider(title,options_ope,"je ne sais pas",key="slide")
#     st.sidebar.table(df_pres)

    

elif False:#st.session_state.cur_page == 404:
        
    bonus, left_column, right_column = st.columns([1,3,3]) # ou 3 ?
    
    # i = left_column.selectbox("Choisissez le match à afficher",
    #                  list(range(1,N_tr+1)))
    i=5
    mode = st.session_state.aide
    # st.write(mode)

    with bonus:
        
        if mode == "neutre":
            st.button("Recommandation IA",on_click=show_xpo)
            mode = st.session_state.aide
    ajust = 0
        # winner = 'rouge' if labels.iloc[(i+ajust-1)%N_tr] else 'bleue'
        # st.title("Équipe gagnante : " + winner)
        # predi = st.session_state.reponses_num.loc[i+4,'rep']
        # st.title("Votre pronostic : " + predi)
        # exp = st.expander('explication IA',expanded=True)   
    if mode != "neutre":
        mode = bonus.radio("Mode de l'interface", ["confiance","voisin","math",
                            "texte"],key='aide',on_change=record)
        # st.session_state.aide = mode
        with right_column:
            col = right_column
            if mode == "bar-SHAP":
                display_shap((i+ajust)%N_tr,sv,col=col)
            elif mode == "pastilles":
                display_pastilles((i+ajust)%N_tr,sv,n=4)
            elif mode == "voisin":
                display_nearest((i+ajust)%N_tr,near1,ann=ann1,col=col)#,df.loc['moy']
            elif mode == 'texte':
                display_rule((i+ajust)%N_tr,df,pred,col=col)
            elif mode == "math":
                display_shap((i+ajust)%N_tr,lime,col=col,plot='flot')
                
            display_score((i+ajust)%N_tr,pred,col=col,wording=2)
            # mode = st.radio("Mode de l'interface", ["confiance","voisin",
            #                          "math","texte"],   )
            # st.session_state.aide = mode
        # else:
    bonus.button("Suite>>>",on_click=nextpage)
    
    left_column.title("Match n° osef")
    display_match((i+ajust-1)%N_tr,df,col=left_column)
    title = "ce slider ne prends pas de décision"
    left_column.select_slider(title,options_ope,"je ne sais pas",key="slide")
    st.sidebar.table(df_pres)
    
#%%

def dec_by_checkboxes(navig):
    if 'red' not in st.session_state:
        st.session_state.red = 0
        st.session_state.blue = 0
    def ChangeRed():
        st.session_state.red,st.session_state.blue = 1,0
    def ChangeBlue():
        st.session_state.red,st.session_state.blue = 0,1
    
    # with left_column:
    #     baril = st.container()
    _, col1, col2, _ = st.columns([2,3,3,6])
    # col1, col2 = baril.columns([1,1])
    with col2:
        a = st.checkbox("l’équipe rouge va vraisemblablement gagner", value=st.session_state.red, on_change=ChangeRed)
    with col1:
        b = st.checkbox("l’équipe bleue va vraisemblablement gagner", value=st.session_state.blue, on_change=ChangeBlue)
    
    dis = (st.session_state.red == st.session_state.blue)
    col1.write(st.session_state.blue)
    col2.write(st.session_state.red)
    st.write(dis)
    r1 = None
    if not dis:
        r1 = 1 if st.session_state.red else 0
    if not dis:
        bonus.button("Suite>>>",on_click=navig)