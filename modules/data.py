# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:27:55 2021

@author: samcl
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import OLSInfluence
import statsmodels.api as sm

class data():
    def descriptives(self, columns):
        descriptive_df = pd.DataFrame(columns = ['Feature', 'Mean', 'Minimum', 'Maximum', 'Range', 'SD', 'Quant1','Quant3'])
        print(descriptive_df.shape)
        i = 0
        for col in columns:
            
            mean = self.df[col].mean()
            minimum, maximum = self.df[col].min(), self.df[col].max()
            Range = maximum - minimum
            sd = self.df[col].std()
            q1, q3 = self.df[col].quantile(q=.25, interpolation='midpoint'), self.df[col].quantile(q=.75, interpolation='midpoint')
            
            descriptive_df.loc[i] = [col, mean, minimum, maximum, Range, sd, q1, q3]
            
            i += 1
            
        print(descriptive_df) 
    
    #create dummy variables
    def dummy(self, columns):
        for col in columns:
            dummy = pd.get_dummies(self.df[col])
            self.df = pd.concat([self.df, dummy], axis = 1)
    
    #downloads data
    def getData(self, filename):
        df = pd.read_csv(filename, encoding = 'unicode_escape')
        
        #remove non functioning days
        self.df = df[df['Functioning Day'] == 'Yes'] 
            
    def influence(self, model):
        influence = model.get_influence()
        cooks_d = influence.cooks_distance
        
        return cooks_d
    
    
    def removePoints(self, df, model, dv, threshold = .01):
        curR2adj = model.rsquared_adj
        newR2adj = 1
        while newR2adj - curR2adj > threshold:
            df['Cooks Distance'] = data().influence(model)[0]
            df['Studentized Residual'] = model.outlier_test()['student_resid']
            n = len(df)
            df_new = df.loc[df['Cooks Distance'] <= 4 /n ]
            df_new = df_new.loc[df['Studentized Residual'] >= -4]
            df_new = df_new.loc[df['Studentized Residual'] <= 4]
            
            df_new = df_new.loc[:, ~df_new.columns.isin(['Cooks Distance', 'Studentized Residual'])]
                    
            #create separate dfs for dv and iv
            df_X = df_new.loc[:, ~df_new.columns.isin([dv])] #features df
            df_Y = df_new[dv] #dv df
                        
            model = sm.OLS(df_Y,df_X).fit()
            
            newR2adj, curR2adj = model.rsquared_adj, newR2adj
            df = df_new
            
        return df
    
    def rename(self, colDict):
        for key in colDict.keys():
            self.df.rename(columns={key:colDict[key]}, inplace = True)
    
    #removes trailing units in column name
    def renameCol(self):
        column_dict = {}
        
        for col in self.df:
            if '(' in col:
                new_name = col[:col.find('(')]
                column_dict[col] = new_name.strip()    
            else:
                column_dict[col] = col.strip()

        self.df.rename(columns=column_dict, inplace = True)
    
    def returnDF(self):
        return self.df
    
    def setDF(self, df):
        self.df = df
    
    
    def student_residual(self, model):
        stud = model.outlier_test()
        print(stud)
    
    #transform variable
    def transform(self, x, transform_type, colName):
        if transform_type.lower() == 'log':
            self.df[colName] = np.log(x)
        if transform_type.lower() == 'sqrt':
            self.df[colName] = np.sqrt(x)
    
    #calculate and remove variables based on VIF
    def vif(self, df, dv):
        df = df.drop(columns = dv)
        drop_col = []
        
        while True:
            df_vif = pd.DataFrame()
            df_vif['Column'] = df.columns
            df_vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
            
            max_val = df_vif['VIF'].max()
            
            if max_val >= 5.0:
                temp_df = df_vif[df_vif['VIF'] == max_val]
                temp_col = temp_df['Column'].iloc[0]
                print(temp_col + ' was removed')
                df = df.drop(columns = [temp_col], axis = 1)
                drop_col.append(temp_col)
            elif max_val < 5:
                break
            
        return drop_col
    


    