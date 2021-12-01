# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:45:44 2021

@author: samcl
"""



#import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

class plots():
    
    def histogram(self, x, num_bins):
        mu = np.mean(x)
        sd = np.std(x)
        
        fig, ax = plt.subplots()
        
        # the histogram of the data
        n, bins, patches = ax.hist(x, num_bins, density=True)

        # add a normal distribution curve
        y = ((1 / (np.sqrt(2 * np.pi) * sd)) *
             np.exp(-0.5 * (1 / sd * (bins - mu))**2))
        ax.plot(bins, y, '--')
        
        '''
        ax.set_xlabel('Bikes Rented')
        ax.set_ylabel('Probability density')
        ax.set_title(r'Histogram of IQ: $\mu=100$, $\sd=15$')
        '''
        
        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        plt.show()
    
    def scatterplot(self, x, y):
        plt.plot(x, y, 'o', color='black')
        
        