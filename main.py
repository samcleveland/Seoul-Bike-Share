# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:10:05 2021

@author: samcl
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from modules.data import *

filename = '../Seoul-Bike-Share/SeoulBikeData.csv'
dv = 'Rented Bike Count'

df = data().getData(filename)
df = data().renameCol(df)


columns = df.columns.tolist()
#columns.remove(dv)

#independent variables
iv = data[columns] 

# set label equal to last column
label = data[dv] 

