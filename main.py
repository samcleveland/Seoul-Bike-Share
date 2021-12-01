# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:10:05 2021

@author: samcl
"""

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
from modules.data import *
from modules.plots import *

filename = '../Seoul-Bike-Share/SeoulBikeData.csv'
dv = 'Rented Bike Count'
dummy_var = ('Hour','Seasons','Holiday')

df = data().getData(filename)
df = data().renameCol(df)

#create dummy variables
df = data().dummy(df, dummy_var)





