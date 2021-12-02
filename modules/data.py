# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:27:55 2021

@author: samcl
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

class data():
    def dummy(self, df, columns):
        for col in columns:
            dummy = pd.get_dummies(df[col])
            df = pd.concat([df, dummy], axis = 1)
            
        return df
    
    def getData(self, filename):
        df = pd.read_csv(filename, encoding = 'unicode_escape')
        
        #remove non functioning days
        df = df[df['Functioning Day'] == 'Yes'] 
        
        return df
    
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
    
    def transform(self, x, transform_type):
        if transform_type.lower() == 'log':
            return np.log(x)
        if transform_type.lower() == 'sqrt':
            return np.sqrt(x)
    

    