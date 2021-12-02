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
iv_col = ['Temperature', 'Humidity', 'Wind speed', 'Visibility', 'Dew point temperature', 'Solar Radiation', 'Rainfall', 'Snowfall' ]
drop_dummy = (23, 'No Holiday','Autumn') #select which dummy variables will be considered base


df = data().getData(filename)
df = data().renameCol(df)


#analyze dependent variable for normal distribution
plots().histogram(np.array(df[dv]), 40, ['Bikes Rented', 'Frequency'])

#SquareRoot transform to get better fit for dependent variable
df['Sqrt Bikes Rented'] = data().transform(np.array(df[dv]), 'sqrt')

dv = 'Sqrt Bikes Rented'

plots().histogram(np.array(df[dv]), 40, [dv, 'Frequency'])



#print scatterplots
for col in iv_col:
    plots().scatterplot(df, col, dv)
    


#create dummy variables
df = data().dummy(df, dummy_var)

#drop base dummy variable
for col in drop_dummy:
    df = df.drop(col, axis = 1)

#create dataframe for correlation matrix
corr_df = pd.concat([df[dv], df[iv_col]], axis = 1)

plots().correlation(corr_df)

data().vif(corr_df)