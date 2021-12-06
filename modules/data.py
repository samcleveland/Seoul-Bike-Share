# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:27:55 2021

@author: samcl
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor

class data():
    def descriptives(self, df, columns):
        descriptive_df = pd.DataFrame(columns = ['Feature', 'Mean', 'Minimum', 'Maximum', 'Range', 'SD', 'Quant1','Quant3'])
        print(descriptive_df.shape)
        i = 0
        for col in columns:
            
            mean = df[col].mean()
            minimum, maximum = df[col].min(), df[col].max()
            Range = maximum - minimum
            sd = df[col].std()
            q1, q3 = df[col].quantile(q=.25, interpolation='midpoint'), df[col].quantile(q=.75, interpolation='midpoint')
            
            descriptive_df.loc[i] = [col, mean, minimum, maximum, Range, sd, q1, q3]
            
            i += 1
            
        print(descriptive_df)
    
    
    #create dummy variables
    def dummy(self, df, columns):
        for col in columns:
            dummy = pd.get_dummies(df[col])
            df = pd.concat([df, dummy], axis = 1)
            
        return df
    
    #downloads data
    def getData(self, filename):
        df = pd.read_csv(filename, encoding = 'unicode_escape')
        
        #remove non functioning days
        df = df[df['Functioning Day'] == 'Yes'] 
        
        return df
    
    #removes trailing units in column name
    def renameCol(self, df):
        column_dict = {}
        
        for col in df:
            if '(' in col:
                new_name = col[:col.find('(')]
                column_dict[col] = new_name.strip()    
            else:
                column_dict[col] = col.strip()

        df.rename(columns=column_dict, inplace = True)
        return df
    
    #transform variable
    def transform(self, x, transform_type):
        if transform_type.lower() == 'log':
            return np.log(x)
        if transform_type.lower() == 'sqrt':
            return np.sqrt(x)
    
    #calculate and remove variables based on VIF
    def vif(self, df, dv):
        df = df.drop(columns = dv)
        
        while True:
            df_vif = pd.DataFrame()
            df_vif['Column'] = df.columns
            df_vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
            
            max_val = df_vif['VIF'].max()
            
            if max_val >= 10.0:
                temp_df = df_vif[df_vif['VIF'] == max_val]
                temp_col = temp_df['Column'].iloc[0]
                print(temp_col + ' was removed')
                df = df.drop(columns = [temp_col], axis = 1)
                print(df.columns)
            elif max_val < 10:
                break
        
        return df
    


    