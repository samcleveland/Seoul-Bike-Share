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
import os
from sklearn.model_selection import train_test_split

os.chdir('C:/Users/samcl/Documents/GitHub/Seoul-Bike-Share')
from modules.data import *
from modules.plots import *

filename = '../Seoul-Bike-Share/SeoulBikeData.csv'
dv = 'Rented Bike Count'
dummy_var = ('Hour','Seasons','Is Holiday')
iv_col = ['Temperature', 'Humidity', 'Wind speed', 'Visibility', 'Dew point temperature', 'Solar Radiation', 'Rainfall', 'Snowfall' ]
drop_dummy = (23, 'No Holiday','Autumn') #select which dummy variables will be considered base


df = data().getData(filename)

df.rename(columns={'Holiday':'Is Holiday'}, inplace = True)

df = data().renameCol(df)


#Create DF with descriptive statistics of IV
descriptive_df = data().descriptives(df, iv_col)


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

df_1 = df.loc[:, ~df.columns.isin(['Date', 'Hour', 'Rented Bike Count', 'Seasons', 'Is Holiday', 'Functioning Day'])]

#need to clean up the data first
#removes variables with multicollinearity
df_1 = df_1.loc[:, ~df_1.columns.isin([data().vif(df_1, dv)])]


#create separate dfs for dv and iv
df_1_x = df_1.loc[:, ~df_1.columns.isin([dv])]
df_1_y = df_1[dv]


# Split data into training and testing datasets
df_1_x_train, df_1_x_test, df_1_y_train, df_1_y_test = train_test_split (X, y, test_size=0.25, random_state=153926)

#split data into training and testing

