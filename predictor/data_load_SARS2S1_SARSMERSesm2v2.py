#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 23:08:05 2025

@author: user
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


path = '../data/'
file_name = 'checkpoint-600_SARS2S1embedding_SARSMERSesm2.pkl'

def loading_data(path = path):
    
    label_file2 = 'df_metadata_forS1_SARS2_testing.csv'
    model_name = 'SARS2S1_checkpoint-600_SARSMERSesm2'   
    
    indexlst_test, labellst_test, variantlst_test = [], [], []
    for file in os.listdir(path):
        if file.endswith(label_file2):
            df = pd.read_csv (path + file, index_col = 'seqID') 
            # print (df.shape)
            indexlst_test = df.index.tolist()
            labellst_test = df.label.tolist()
       

    with open(path + file_name, 'rb') as f:
        data2 = pickle.load(f)
        array2 = np.array(data2)[:,:245, :]
        sample_num2 = array2.shape[0]
        reshaped_array2 = array2.reshape(sample_num2,-1)
        print (reshaped_array2.shape)
    
    X = reshaped_array2.reshape((sample_num2, 560, 560))
    print (X.shape)
    
    X = X[25000:30000, :, :]
    
    ylst = labellst_test
    ylst = ylst[25000:30000]
    
    y = np.array(ylst)

    
    return X, y


loading_data()


