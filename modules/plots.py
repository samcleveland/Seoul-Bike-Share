# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:45:44 2021

@author: samcl
"""


#import numpy as np
#import pandas as pd

import matplotlib.pyplot as plt

class plots():
    
    def scatterplot(self, x, y):
        plt.plot(x, y, 'o', color='black')
        
        