# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:02:25 2018

@author: CXJ
"""

import logging
import pickle
import itertools
import timeit
import pandas as pd
import math
import os

if __name__ == "__main__":
    #data1 = pd.read_csv('./../datasets/best.3best.100.1',sep=' ',usecols=[0,1],header=None)
    #data1 = pd.read_csv('./../datasets/best.3AC.100.1',sep=' ',usecols=[0,1],header=None)
    #data1 = pd.read_csv('./../datasets/best.3BC.100.1',sep=' ',usecols=[0,1],header=None)
    #data1 = pd.read_csv('./../datasets/best.3CD.100.1',sep=' ',usecols=[0,1],header=None)
    data1 = pd.read_csv('./../datasets/best.3DE.100.1',sep=' ',usecols=[0,1],header=None)

    comprision = data1.values
    print type(data1)
    #data1.to_csv('best_AB.txt',index=False,sep=' ',header=None)
    #data1.to_csv('best_AC.txt',index=False,sep=' ',header=None)
    #data1.to_csv('best_BC.txt',index=False,sep=' ',header=None)
    #data1.to_csv('best_CD.txt',index=False,sep=' ',header=None)
    data1.to_csv('best_DE.txt',index=False,sep=' ',header=None)


    
