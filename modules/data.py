# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:27:55 2021

@author: samcl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

class data():  
    def descriptives(self, columns):
        'produces descriptive statstics'
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
        'turns columns into dummy variables'
        for col in columns:
            dummy = pd.get_dummies(self.df[col])
            self.df = pd.concat([self.df, dummy], axis = 1)
    
    #downloads data
    def getData(self, filename):
        'reads file and creates df'
        df = pd.read_csv(filename, encoding = 'unicode_escape')
        
        #remove non functioning days
        self.df = df[df['Functioning Day'] == 'Yes'] 
        
    def getR2(self):
        'returns r squared value'
        return self.reg.rsquared_adj
            
    def influence(self, model):
        'returns cooks distance'
        influence = model.get_influence()
        cooks_d = influence.cooks_distance
        
        return cooks_d
    
    def fit(self, x, y):
        'fit model'
        new_x = sm.add_constant(x, prepend=True)
        self.reg = sm.OLS(y,new_x).fit()
        
        print(self.reg.summary())
        
        pvalues = self.reg.pvalues

        if max(self.reg.pvalues) > .05:
            self.removeCol([pvalues.idxmax()])
            self.split()
            self.fit(self.df_X, self.df_Y)
        
        return self.reg
    
    def predict(self):
        'predicts the dv on test data set'
        self.predict_model = self.fit(self.df_x_train, self.df_y_train)
        
        self.df_predict = pd.DataFrame()
        
        test_dataset =  sm.add_constant(self.df_x_test, prepend=True)
        
        self.df_predict['Model'] = self.predict_model.predict(test_dataset)
        self.df_predict['Actual'] = self.df_y_test
        
        return self.df_predict
    
    
    def removePoints(self):
        'removes outliers and influential points from dataset'
        self.df['Cooks Distance'] = self.influence(self.reg)[0]
        self.df['Studentized Residual'] = self.reg.outlier_test()['student_resid']
        n = len(self.df)
        self.df_new = self.df.loc[self.df['Cooks Distance'] <= 4 /n ]
        self.df_new = self.df_new.loc[self.df['Studentized Residual'] >= -4]
        self.df_new = self.df_new.loc[self.df['Studentized Residual'] <= 4]
        
        self.df_new = self.df_new.loc[:, ~self.df_new.columns.isin(['Cooks Distance', 'Studentized Residual'])]
                
        #create separate dfs for dv and iv
        self.df_X = self.df_new.loc[:, ~self.df_new.columns.isin([self.dv])] #features df
        self.df_Y = self.df_new[self.dv] #dv df
                    
        self.fit(self.df_X,self.df_Y)
        
        self.df = self.df_new
            
        return self.df
    
    def rename(self, colDict):
        'renames columns based on dictionary values'
        for key in colDict.keys():
            self.df.rename(columns={key:colDict[key]}, inplace = True)
    
    #removes trailing units in column name
    def renameCol(self):
        'remove extraneous wording and spaces from column names'
        column_dict = {}
        
        for col in self.df:
            if '(' in col:
                new_name = col[:col.find('(')]
                column_dict[col] = new_name.strip()    
            else:
                column_dict[col] = col.strip()

        self.df.rename(columns=column_dict, inplace = True)
    
    def returnDF(self):
        'returns data frame'
        return self.df
    
    def removeCol(self, cols):
        'deletes column from dataframe'
        self.df = self.df.loc[:, ~self.df.columns.isin(cols)]
        
        return self.df
            
    def setDF(self, df):
        'sets df'
        self.df = df
    
    def setDV(self, dv):
        'sets dependent variable'
        self.dv = dv
        
    def split(self):
        'splits dataframe into iv and dv'
        self.df_X = self.df.loc[:, ~self.df.columns.isin([self.dv])]
        self.df_Y = self.df[self.dv]
    
    def student_residual(self, model):
        'print studentized residulas'
        stud = model.outlier_test()
        print(stud)
        
    def train_test(self, seed):
        'split df into train and test'
        self.df_x_train, self.df_x_test, self.df_y_train, self.df_y_test = train_test_split(self.df_X, self.df_Y, test_size=0.25, random_state=seed)
    

    def transform(self, x, transform_type, colName):
        'transforms dv'
        if transform_type.lower() == 'log':
            self.df[colName] = np.log(x)
        if transform_type.lower() == 'sqrt':
            self.df[colName] = np.sqrt(x)
    
    #calculate and remove variables based on VIF
    def vif(self, threshold):
        'remove columns with VIF larger than threshold'
        df = self.df.drop(columns = self.dv)
        drop_col = []
        
        while True:
            df_vif = pd.DataFrame()
            df_vif['Column'] = df.columns
            df_vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
            
            max_val = df_vif['VIF'].max()
            
            if max_val >= threshold:
                temp_df = df_vif[df_vif['VIF'] == max_val]
                temp_col = temp_df['Column'].iloc[0]
                print(temp_col + ' was removed')
                df = df.drop(columns = [temp_col], axis = 1)
                drop_col.append(temp_col)
            else:
                break
        
        self.removeCol(drop_col)
    


    