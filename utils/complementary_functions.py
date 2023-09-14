#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:54:41 2022

@author: Corentin.Boidot
"""

import pandas as pd
import matplotlib.pyplot as plt







rond = lambda f,g: lambda x: f(g(x))


def df_f(f,*args,**kwargs):
    """
    Fonction conçue comme un wrapper, en vue d'être appliquée aux nombreuses 
    méthodes de sklearn qui renvoient un np.array au lieu du pd.DataFrame
    qu'on leur donne en entrée
    Parameters
    ----------
    f : function: pd.DataFrame -> np.array
        Méthode à décorer.
    X : pd.DataFrame
        argument principal de la méthdoe.
    Returns
    -------
    f: pd.DataFrame -> pd.DataFrame, ou pd.Series
        Résultat de f au bon format.
    @author Corentin Boidot, corentinboidot@free.fr
    """
    def f_(X):
        res = f(X,*args,**kwargs)
        c,i = None, None
        if res.shape[0]==X.shape[0]:
            i = X.index
        if len(res.shape)==1:
            return pd.Series(res, index=i)
        if res.shape[1]==X.shape[1]:
            c = X.columns
        return pd.DataFrame(res,columns=c,index=i)
    return f_
    

def df_fx(f,X,*args,**kwargs):
    '''
    df_f appliquée à X.
    Parameters
    f : function: pd.DataFrame -> np.array
        Méthode à décorer.
    X : pd.DataFrame
        argument principal de la méthdoe.
    Returns
    pd.DataFrame ou pd.Series
    '''
    return df_f(f,*args,**kwargs)(X)


def err_indices(ytrue, ypred, err_type="all", n=1):
    """
    Cette fonction est là pour donner des exemples d'erreurs à observer.
    Parameters
    ytrue : pandas.Series
    ypred : /!\ pandas.Series /!\ you should use df_fx ;)
    err_type : str, optional -> "all" "FP" or "FN"
    n : int, optional : number of examples of each kind
    Returns
    l : list of indices (relative to ytrue.iloc)
        @protips: err_indices(1-ytrue,ypred) vous donne des exemples pour des
        bonnes détections !
    """
    l = []
    n_FP = 0
    n_FN = 0
    n_lim = 0
    i = 0
    while n_lim < n and i < ypred.shape[0]:
        if err_type != "FN" and ypred.iloc[i]==1 and ytrue.iloc[i]==0:
            if n_FP<n:
                l.append(i)
            # print(i)
            # print(ytrue.iloc[i])
            n_FP += 1
        if err_type != "FP" and ypred.iloc[i]==0 and ytrue.iloc[i]==1:
            if n_FN<n:
                l.append(i)
            n_FN += 1
        if err_type == "all":
            n_lim = min(n_FN,n_FP)
        else:
            n_lim = max(n_FN,n_FP)
        i += 1
    return l


def error_graph(true_labels, scores, thr=.5, bins=30, n_points=30,log=False,
                en=False,wrap=False):
    """
    Affiche l'histogramme des scores, et celui des erreurs, celles-ci étant 
    déterminées d'après un seuil global.
    Fonction conçue pour des prévalences égales.

    Parameters
    ----------
    true_labels : array-like
    scores : array-like
    thr : float, optional
        Seuil de décision. The default is .5.
    bins : int ou linspace, optional
    The default is 30.
    n_points : int, optional
    The default is 30.
    log : Boolean, optional
    Activate log_view. The default is False.
    """
    x_min, x_max = scores.min(),scores.max()
    # grilles pour les valeurs de score.
    # grid = np.linspace(x_min,x_max,n_points) # pour les densités sup et inf
    ## affichage des distributions de score
    if wrap:
        x_min=.5
        
        fig,ax = plt.subplots()
        _,bins,__ = ax.hist(scores.apply(lambda x: max(x,1-x)),bins=bins,log=log,label="Total", color='grey')
        ax.hist(scores[true_labels!=(scores>thr)],bins=bins,label="Errors", color="red")
        title = "Distribution of displayed scoring" if en else "Histogramme des scores affichés"
    else:
        fig,ax = plt.subplots()
        _,bins,__ = ax.hist(scores,bins=bins,log=log,label="total", color='grey')
        ax.hist(scores[(true_labels==0)&(scores>thr)],
                bins=bins,label="FP", color="red")
        ax.hist(scores[(true_labels==1)&(scores<thr)],
                bins=bins,label="FN", color="violet")
        #plt.gca(axes="equal")
        title = "Distribution of games by ML scoring" if en else "Histogramme des matchs par scores attribués"
    plt.title(title)
    plt.xlabel("ML score" if en else "score ML")
    fig.tight_layout()
    plt.legend()
    plt.show()