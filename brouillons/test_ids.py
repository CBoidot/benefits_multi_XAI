#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:43:53 2023

Testé dans le testenv...
gros fouilli

@author: Corentin.Boidot
"""


import pandas as pd
from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS

from pyarc.qcba.data_structures import QuantitativeDataFrame
import io
import requests

url = "https://raw.githubusercontent.com/kliegr/arcBench/master/data/folds_discr/train/iris0.csv"
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))
cars = mine_CARs(df, rule_cutoff=50)
lambda_array = [1, 1, 1, 1, 1, 1, 1]

quant_dataframe = QuantitativeDataFrame(df)

ids = IDS(algorithm="SLS")
ids.fit(quant_dataframe=quant_dataframe, class_association_rules=cars, lambda_array=lambda_array)

acc = ids.score(quant_dataframe)
