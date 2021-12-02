# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:10:05 2021

@author: samcl

Data provided by UCI
https://archive.ics.uci.edu/ml/machine-learning-databases/00560/
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
iv_col = ('Temperature', 'Humidity', 'Wind speed', 'Visibility', 'Solar Radiation', 'Rainfall', 'Snowfall' )


df = data().getData(filename)
df = data().renameCol(df)


#analyze dependent variable for normal distribution
plots().histogram(np.array(df['Rented Bike Count']), 40, ['Bikes Rented', 'Frequency'])

#SquareRoot transform to get better fit for dependent variable
df['Sqrt Bikes Rented'] = data().transform(np.array(df['Rented Bike Count']), 'sqrt')

plots().histogram(np.array(df['Sqrt Bikes Rented']), 40, ['Sqrt Bikes Rented', 'Frequency'])



#print scatterplots
for col in iv_col:
    plots().scatterplot(df, col, 'Sqrt Bikes Rented')
    

#create dummy variables
df = data().dummy(df, dummy_var)    
