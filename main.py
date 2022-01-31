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
#from statsmodels.formula.api import ols
import statsmodels.api as sm
#from statsmodels.stats.outliers_influence import variance_inflation_factor as vif



os.chdir('C:/Users/samcl/Documents/GitHub/Seoul-Bike-Share')
from modules.data import *
from modules.plots import *

filename = '../Seoul-Bike-Share/SeoulBikeData.csv' #filename of dataset
dv = 'Rented Bike Count' #dependent variable
dummy_var = ('Hour','Seasons','Is Holiday') #list of dummy variables
iv_col = ['Temperature', 'Humidity', 'Wind speed', 'Visibility', 'Dew point temperature', 'Solar Radiation', 'Rainfall', 'Snowfall' ] #independent variables/non-dummy variables
drop_dummy = (23, 'No Holiday','Autumn') #select which dummy variables will be considered base


df = data().getData(filename) #import data

df.rename(columns={'Holiday':'Is Holiday'}, inplace = True) #rename holiday column for clarity

df = data().renameCol(df) #remove certain characters from headings


#Create DF with descriptive statistics of IV
descriptive_df = data().descriptives(df, iv_col)


#analyze dependent variable for normal distribution
plots().histogram(np.array(df[dv]), 40, ['Bikes Rented', 'Frequency'])

#SquareRoot transform to get better fit for dependent variable
df['Sqrt Bikes Rented'] = data().transform(np.array(df[dv]), 'sqrt')

dv = 'Sqrt Bikes Rented' #set new dv name

plots().histogram(np.array(df[dv]), 40, [dv, 'Frequency']) #replot histogram with transformed dv

#print scatterplots to check linearity
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

#removes variables with VIF>10
df_1 = df_1.loc[:, ~df_1.columns.isin([data().vif(df_1, dv)])]


#create separate dfs for dv and iv
df_1_X = df_1.loc[:, ~df_1.columns.isin([dv])] #features df
df_1_Y = df_1[dv] #dv df

#train full model
reg = sm.OLS(df_1_Y,df_1_X).fit()

#remove influential points & outliers
inf = reg.get_influence()
inf.plot_influence()










print(reg.rsquared_adj)

# Train the classifier
#reg.fit(df_1_x, df_1_y)

#full model prediction
full_model = pd.DataFrame()
df_1['full_y_pred'] = reg.predict(df_1_x)


stud_resid = data().student_residual(ol_model)
df_1['Resdiual'] = stud_resid[0]

cooks_d = data().influence(reg)
df_1['cooks_d'] = cooks_d[0]

# Split data into training and testing datasets
df_1_x_train, df_1_x_test, df_1_y_train, df_1_y_test = train_test_split (X, y, test_size=0.25, random_state=153926)



