# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:45:44 2021

@author: samcl
"""



#import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class plots():
    def correlation(self):
        #get the correlations of each feature in the dataset
        corr_matrix = self.df.corr()
        top_corr_features = corr_matrix.index
        plt.figure(figsize=(20,20))
        #plot heat map
        sns.heatmap(self.df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
        
    def setDF(self, df):
        self.df = df
    
    def histogram(self, x, num_bins, title):
        mu = np.mean(x)
        sd = np.std(x)
                
        fig, ax = plt.subplots()
        
        # the histogram of the data
        n, bins, patches = ax.hist(x, num_bins, density=True)

        # add a normal distribution curve
        y = ((1 / (np.sqrt(2 * np.pi) * sd)) *
             np.exp(-0.5 * (1 / sd * (bins - mu))**2))
        ax.plot(bins, y, '--')
        
        ax.set_xlabel(title[0])
        ax.set_ylabel(title[1])
        ax.set_title(title[1] + ' of ' + title[0])
        
        
        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        plt.show()
    
    def scatterplot(self, df, x, y):
        fig, ax = plt.subplots()        
        plt.plot(np.array(df[x]), np.array(df[y]), 'o', color='black')
        
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(y + ' by ' + x)
        
        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        plt.show()
        
        