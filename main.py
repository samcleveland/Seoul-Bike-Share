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


main_df = data() #create instance of Data will full dataset
main_df.getData(filename) #import data

main_df.rename({'Holiday':'Is Holiday'}) #rename holiday column for clarity

main_df.renameCol() #remove special characters from headings

#Create DF with descriptive statistics of IV
main_df.descriptives(iv_col)  #try to create a more exhibity looking thing

getPlot = plots()

#analyze dependent variable for normal distribution
getPlot.histogram(np.array(main_df.df[dv]), 40, ['Bikes Rented', 'Frequency'])

#SquareRoot transform to get better fit for dependent variable
main_df.transform(np.array(main_df.df[dv]), 'sqrt','Sqrt Bikes Rented')

dv = 'Sqrt Bikes Rented' #set new dv name
main_df.setDV(dv)

getPlot.histogram(np.array(main_df.df[dv]), 40, [dv, 'Frequency']) #replot histogram with transformed dv

#print scatterplots to check linearity
for col in iv_col:
    getPlot.scatterplot(main_df.df, col, dv)
    
#create dummy variables
main_df.dummy(dummy_var)

check_df = main_df.df

#drop base dummy variable
for col in drop_dummy:
    main_df.df = main_df.df.drop(col, axis = 1)

#set dataframe in getPlot
getPlot.setDF(pd.concat([main_df.df[dv], main_df.df[iv_col]], axis = 1))

#print correlation of independent variables 
getPlot.correlation()

#remove unnecessary columns
main_df.removeCol(['Date', 'Hour', 'Rented Bike Count', 'Seasons', 'Is Holiday', 'Functioning Day'])

#removes variables with VIF>10
main_df.vif(10)

main_df.df.columns

#create separate dfs for dv and iv
main_df.split()

#train full model
main_df.fit()

#remove influential points & outliers
inf = main_df.reg.get_influence()
inf.plot_influence()






df_1 = data().removePoints(df_1, reg, dv, .01)

df_test = pd.DataFrame()
df_test["cooks d"] = data().influence(reg)

#create separate dfs for dv and iv
df_1_X = df_1.loc[:, ~df_1.columns.isin([dv])] #features df
df_1_Y = df_1[dv] #dv df

#train full model excluding outliers
reg = sm.OLS(df_1_Y,df_1_X).fit()
print(reg.rsquared_adj)

# Split data into training and testing datasets
df_1_x_train, df_1_x_test, df_1_y_train, df_1_y_test = train_test_split (df_1_X, df_1_Y, test_size=0.25, random_state=153926)

#create training model
train_model = sm.OLS(df_1_y_train, df_1_x_train).fit()
print(train_model.summary())


#df_predict = pd.DataFrame()


#df_predict['Model'] = train_model.predict(df_1_x_test)



