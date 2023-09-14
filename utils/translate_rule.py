#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 13:13:43 2023

@author: Corentin.Boidot
"""

import numpy as np
import pandas as pd

import pickle
import re


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

def translate_ft(f):
    # if feat_name[:-3] == 'ood': # pour Blood
    #     team = 'rouge' if value_desc == 1 else 0
    #     descr.append(desc_format[feat_name].format(team))
    # else:
    #     team_desc = "bleue"
    #     if feat_name[0:3] == 'red':
    #         team_desc = 'rouge'
    #         feat_name = 'blue' + feat_name[3:]
    #     descr.append(desc_format[feat_name].format(team_desc, value_desc))
    # if fi > 0:
    #     xp_sense = "r"
    # else:
    #     xp_sense = "b"
    return f

def transform_value(x,ft):
    if ft in cont_c:
        with open('robust_scaler.pkl','rb') as f:
            rs = pickle.load(f)
        # rs = ct.transformers[1][1]
        img = pd.DataFrame(np.zeros((1,16)),columns=cont_c)
        img[ft] = x
        ante = rs.inverse_transform(img).astype(int)
        x = dict(zip(cont_c,ante[0]))[ft]
    else:
        print("WTF")
        print(ft)
        print
        x = round(float(x),1)
    return x

def translate_rule_(r):
    # r = re.sub('blue[a-zA-Z]*',translate_ft,r) # 
    # r = re.sub('red[a-zA-Z]*',translate_ft,r)
    # r = re.sub('[+-]?([0-9]*[.])?[0-9]+','pi',r)
    # r = re.sub(' and ',' et ',r)
    # r = re.sub(' > ',' est supérieur à ',r)
    # r = re.sub(' <= ',' est inférieur à ',r)
    rl = re.split(" and ",r)
    resl = []
    for er in rl:
        er = "".join(re.split(' +',er)) # nique toi
        ft,_,x = re.split('>|(<=)',er)
        if '>' in er:
            resl.append(translate_ft(ft) +
                        ' est supérieur à ' +
                        str(transform_value(x,ft)))
        elif '<=' in er:
            resl.append(translate_ft(ft) +
                        ' est inférieur à ' +
                        str(transform_value(x,ft)))
        res = " et ".join(resl)
    return res

def translate_rule(r_,target,wording=1):
    fact = translate_rule_(r_[0])
    if wording==1:
        res = "Si "+fact+" alors "
        ending = "l'équipe {} gagnera."
    if wording==2:
        res = "Comme "+fact+", "
        ending = "l'équipe {} devrait certainement gagner."
    target = {"blue":'bleue',"red":'rouge'}[target]
    return res + ending.format(target)



# "Si l'équipe rouge a vaincu 1,53 dragons ou plus et..."


