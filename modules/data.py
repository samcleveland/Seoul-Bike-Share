# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:27:55 2021

@author: samcl
"""

import pandas as pd

class data():
    def getData(self, filename):
        df = pd.read_csv(filename, encoding = 'unicode_escape')
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