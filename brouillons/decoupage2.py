#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 09:51:59 2022

exécuté avec cbenv1.
Majoritairement copié collé de decoupage.

Mon idée est de ne plus sélectionner en conformité avec RF (ce qui 
nécessairement, biaiserait les mesures), mais en gardant juste un œil sur
le modèle linéaire, que je juge plus neutre, plus apte à représenter la 
difficulté réelle des données, leur bruit.


@author: Corentin.Boidot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",40)
pd.set_option("display.max_rows",30)

import pickle
from sklearn import linear_model#, LinearRegression, LogisticRegression
from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler, OrdinalEncoder




# with open("sv_vrac_linear.pkl",'rb') as pf:
#     sv_rf = pickle.load(pf)
path_d = '../../these1A/data/'
path_m = '../../these1A/model/'

with open(path_d+"re_Xtrain.pkl",'rb') as pf:
    X_train = pickle.load(pf)
with open(path_d+"re_Xtest.pkl",'rb') as pf:
    X_test = pickle.load(pf)
    
with open(path_d+"re_ytest.pkl",'rb') as pf:
    # j'ai gardé le stockage originel, où 1 = victoire des bleus
    y_test = 1 - pickle.load(pf)

with open(path_m+"re_linear.pkl",'rb') as pf:
    # j'ai gardé le stockage originel, où 1 = victoire des bleus
    reg = pickle.load(pf)

# X_tt = X_test

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
col_transfo = [('categories',
                OrdinalEncoder(categories=[[0,1]]*7,
                               handle_unknown='use_encoded_value',
                               unknown_value=2),
                categ_c),
                ('continues',  
                 RobustScaler(quantile_range=(5,95)), 
                 cont_c)]
c_tr = ColumnTransformer(col_transfo,
         remainder='passthrough', verbose_feature_names_out=False)

## importer df_fx en même temps que ce qui suit...
# tout est dans complementary_functions.py de Exp1-Streamlit
X_tr = c_tr.fit_transform(X_train)
X_tt = df_fx(c_tr.transform,X_test)

y_pred = 1 - df_fx(lambda x: reg.predict_proba(x)[:,1],X_tt)


##############################################################################
#                     Extraction de la séquence de test                      #
##############################################################################


#                                   /!\/!\/!\
# /!\/!\/!\ importer err_indice et error_graph de cbtk.visualization /!\/!\/!\
#                                   /!\/!\/!\
from utils.complementary_functions import error_graph

error_graph(y_test,y_pred,bins=np.linspace(0,1,11))
    
extrait = X_test.sample(frac=1)
y_pred = y_pred.reindex_like(extrait)
y_test = y_test.reindex_like(extrait)
 
y_t_select = y_test.iloc[:80]
y_p_select = y_pred.iloc[:80]
red_y = y_t_select.sum()
if red_y>37 and red_y<43:
    error_graph(y_t_select,y_p_select,bins=np.linspace(0,1,11))
print(red_y)


dec = (y_pred>0.5).apply(int)
df_select = extrait.iloc[:80]



# ######
#    Après plusieurs éxécutions du code précédent, comparé à l'error_graph 
# global, j'ai un extrait qui me convient. (cf protocole décrit ici :
# https://docs.google.com/document/d/1llR3kUU3HoLgkyx8InydLi1xc6UwTAtc639v_es0fvs/edit?pli=1
# Ici, je le mets en ordre avant de l'enregistrer.

# /!\ attention, à partir d'ici tr fait référence à l'user training

iptr = df_select[(y_t_select==1)].iloc[:15].index
intr = df_select[(y_t_select==0)].iloc[:15].index
i_tr = iptr.union(intr)
i_ev = y_t_select.index.difference(i_tr)

# /!\ ici, on a un index trié... a priori, la base de départ (et donc ses 
# index) n'avait pas d'ordre, mais méfiance... /!\

df_tr = df_select.loc[i_tr]
df_ev = df_select.loc[i_ev]

pred_tr = y_pred.loc[i_tr]
label_tr = y_test.loc[i_tr]
error_graph(label_tr,pred_tr,bins=np.linspace(0,1,11))
pred_ev = y_pred.loc[i_ev]
label_ev = y_test.loc[i_ev]
error_graph(label_ev,pred_ev,bins=np.linspace(0,1,11))



# pd.DataFrame([y_test,dec])
# case = pd.DataFrame([y_test,dec])
# case = case.T
# case.columns = ['truth','predicted']
# # case.columns
# # Index(['blueWins', 'Unnamed 0'], dtype='object')



# # Je suis satisfait de mon tirage.

# potential doubled = [450,1110,1289,2164,2866]


# Allez, petit bonus, on renomme

new_order = ['firstBlood', 'blueDragons', 'redDragons', 'blueHeralds',
 'redHeralds', 'blueTowersDestroyed', 'redTowersDestroyed', 'blueWardsPlaced',
 'redWardsPlaced', 'blueWardsDestroyed', 'redWardsDestroyed', 'blueKills', 
 'redKills', 'blueAssists', 'redAssists', 'blueTotalGold', 'redTotalGold',
 'blueTotalExperience', 'redTotalExperience', 'blueTotalMinionsKilled', 
 'redTotalMinionsKilled', 'blueTotalJungleMinionsKilled',
 'redTotalJungleMinionsKilled']

def reframe(df):
    df["blueFirstBlood"] = df.blueFirstBlood.apply(
        lambda x: "blue" if x==1 else "red"
        )
    df = df.rename(columns={'blueFirstBlood':'firstBlood'})
    # autrefois, le réordonnement était fait dans main_expe
    # maintenant, c'est ici
    df = df[new_order]
    return df

df_tr = reframe(df_tr)
df_ev = reframe(df_ev)

moyennes = pd.DataFrame(X_train.mean(),columns=["moy"])
moyennes = reframe(moyennes.T)
moyennes = moyennes.apply(lambda x: round(x,2) if str(x.iloc[0])!=x.iloc[0] else x)
moyennes.loc['moy','firstBlood'] = "50% blue"
# # Je repasse par paper_draft1.py pour avoir les SV de ces données dans l'ordre.
# # Et je compare !


df_tr.loc['moy'] = moyennes.loc['moy']
df_ev.loc['moy'] = moyennes.loc['moy']


with open("Exp1-Streamlit/selection_tr.pkl",'wb') as pf:
      pickle.dump(df_tr,pf)
with open("Exp1-Streamlit/selection_ev.pkl",'wb') as pf:
      pickle.dump(df_ev,pf)
with open("Exp1-Streamlit/label_tr.pkl",'wb') as pf:
      pickle.dump(label_tr,pf)
with open("Exp1-Streamlit/label_ev.pkl",'wb') as pf:
      pickle.dump(label_ev,pf)
with open("Exp1-Streamlit/linear_tr.pkl",'wb') as pf:
      pickle.dump(pred_tr,pf)
with open("Exp1-Streamlit/linear_ev.pkl",'wb') as pf:
      pickle.dump(pred_ev,pf)



## le 16 mars, je veux remélanger l'éval pour que les 10 premiers cas soient
#  plus équilibrés.
with open("Exp1-Streamlit/selection_ev.pkl",'rb') as pf:
      df_select = pickle.load(pf)
with open("Exp1-Streamlit/label_ev.pkl",'rb') as pf:
      label = pickle.load(pf)


# repeat
df_select = df_select.sample(frac=1)
diff = df_select.redTotalGold - df_select.blueTotalGold
diff.iloc[:10].hist() # je veux que ça soit équilibré

# (and fade)

label = label.loc[df_select.drop('moy').index] #.iloc[:10]


with open("these1A/model/re_RF.pkl",'rb') as pf:
    rf = pickle.load(pf)
with open(path_m+"column_transformer", 'rb') as f:
    c_tr = pickle.load(f)
with open(path_d+"re_"+"Xtest.pkl",'rb') as pf:
    X_test = pickle.load(pf)
    
rf.predict_proba(c_tr.transform(X_test.loc[df_select.iloc[:10].index]))
# Out[285]: 
# array([[0.835, 0.165],
#        [0.595, 0.405],
#        [0.585, 0.415],
#        [0.215, 0.785],
#        [0.315, 0.685],
#        [0.235, 0.765],
#        [0.72 , 0.28 ],
#        [0.83 , 0.17 ],
#        [0.695, 0.305],
#        [0.13 , 0.87 ]])

with open("Exp1-Streamlit/selection_ev.pkl",'wb') as pf:
      pickle.dump(df_select,pf)
with open("Exp1-Streamlit/label_ev.pkl",'wb') as pf:
      pickle.dump(label,pf)






# xpo_ult.data = selection_ultime
# et on resauvegarde les SV, la selection, ainsi que les moyennes à causes du 
# renommage   



# with open("/home/Corentin.Boidot/Documents/codes/git-project/these-corenti/model/lol2_RF.pkl",'rb') as pf:
#     reg = pickle.load(pf)

# import shap
# path_d = "/home/Corentin.Boidot/Documents/codes/git-project/these-corenti/data/lol2_"
# with open(path_d+"Xtrain.pkl",'rb') as pf:
#     X_train = pickle.load(pf)
    
#     # ... il faut refaire le preprocessing pour avoir X_tr
    
# xpr = shap.LinearExplainer(reg,X_tr)
# # attention à préprocesser !
# # c'est rigolo, sans le preprocess, il n'y avait as besoin de check_add :)
# xpo_ult = xpr(df_fx(c_tr.transform,selection_ultime),check_additivity=False)[:,:,1]
# #sale bête
# xpo_ult.values = -xpo_ult.values
# xpo_ult.base_values = 1-xpo_ult.base_values



# with open("/home/Corentin.Boidot/Documents/these-corentin/Exp1-Streamlit/shapley_values.pkl",'wb') as pf:
#       pickle.dump(xpo_ult,pf)
# # et là normalement, c'est fini, j'ai créé les bons fichiers, en cohérence.





# ### choix de la partie d'exemple

# N_cas = 1

# extrait = X_test.drop(selection_ultime.index).sample(frac=1)
# y_pred_ = y_pred.drop(selection_ultime.index).reindex_like(extrait)
# y_test_ = y_test.drop(selection_ultime.index).reindex_like(extrait)
# dec = (y_pred_>0.5).apply(int)
# indices = []
# indices += err_indices(1-y_test_, dec, err_type="FP", n=N_cas) # TP
# indices += err_indices(1-y_test_, dec, err_type="FN", n=N_cas) # TN

# # je refais jusqu'à tomber sur des y_pred qui me plaisent -> (.11-.29) et 
# # (.51-.69 ou .81-.89)

# # y_pred.iloc[indices]
# # Out[62]: 
# # 3885    0.860
# # 161     0.235
# extrait = X_test.loc[[2414,9839]]

# y_p_ex = y_pred.loc[[2414,9839]]

# extrait["blueFirstBlood"] = extrait.blueFirstBlood.apply(
#     lambda x: "blue" if x==1 else "red"
#     )
# extrait = extrait.rename(columns={'blueFirstBlood':'firstBlood'})


# with open("example_data.pkl",'wb') as pf:
#      pickle.dump(extrait,pf)
# with open("example_score.pkl",'wb') as pf:
#      pickle.dump(y_p_ex,pf)

# xpo_ult = xpr(df_fx(c_tr.transform,extrait.iloc[1:]),check_additivity=False)[:,:,1]
# xpo_ult.values = -xpo_ult.values
# xpo_ult.base_values = 1-xpo_ult.base_values

# with open("example_sv.pkl",'wb') as f:
#     pickle.dump(xpo_ult,f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# #####

# # Un passage de paper1_draft que j'ai souvent éxécuté...


# ord_c = ['blueFirstBlood',
#          'blueDragons',
#          'blueHeralds',
#          'blueTowersDestroyed',
#          'redDragons',
#          'redHeralds',
#          'redTowersDestroyed',
#          'blueWardsPlaced',
#          'blueWardsDestroyed',
#          'blueKills',
#          'blueAssists',
#          'blueTotalGold',
#          'blueTotalExperience',
#          'blueTotalMinionsKilled',
#          'blueTotalJungleMinionsKilled',
#          'redWardsPlaced',
#          'redWardsDestroyed',
#          'redKills',
#          'redAssists',
#          'redTotalGold',
#          'redTotalExperience',
#          'redTotalMinionsKilled',
#          'redTotalJungleMinionsKilled']
# categ_c = ord_c[:7]
# cont_c = ord_c[7:]
# col_transfo = [('categories',
#                 OrdinalEncoder(categories=[[0,1]]*7,
#                                handle_unknown='use_encoded_value',
#                                unknown_value=2),
#                 categ_c),
#                 ('continues',  
#                  RobustScaler(quantile_range=(5,95)), 
#                  cont_c)]    



# X_train, X_test = X_train[ord_c], X_test[ord_c] # au cas où...
# c_tr = ColumnTransformer(col_transfo,
#          remainder='passthrough', verbose_feature_names_out=False)

# X_tr = df_fx(c_tr.fit_transform,X_train)
# X_tt = df_fx(c_tr.transform,X_test)



# ### le 7 fev, pendant la deuxième série (2,4,6)

# y_tt_choisi = y_test.reindex(dec.index)
# with open("ground_truth.pkl","wb") as f:
#     pickle.dump(y_tt_choisi,f)



### 4 mai 2023
with open("data_cache/rf_tr.pkl",'rb') as pf:
    pred_tr = pickle.load(pf)
with open("data_cache/rf_ev.pkl",'rb') as pf:
    pred_ev = pickle.load(pf)
    
# error_graph(pd.concat([label_tr,label_ev]),pd.concat([pred_tr,pred_ev]),bins=np.linspace(0,1,11),en=True)

true_labels=pd.concat([label_tr,label_ev]).reset_index(drop=True)
scores=pd.concat([pred_tr,pred_ev]).reset_index(drop=True)
scores = scores.loc[true_labels.index]

fig,ax = plt.subplots()
_,bins,__ = ax.hist(scores,bins=bins,label="total", color='grey')
ax.hist(scores[(true_labels==0)&(scores>thr)|(true_labels==1)&(scores<thr)],
        bins=bins,label="errors", color="red")
#plt.gca(axes="equal")
title = "Distribution of games by ML scoring" if en else "Histogramme des matchs par scores attribués"
plt.title(title)
plt.xlabel("ML score" if en else "score ML")
fig.tight_layout()
plt.legend()
plt.show()