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

def loading_data(path = path):
    label_file1 = 'df_parsed_SpikeS1_MERS_SARS_SARS2.csv'
    
    indexlst1, labellst1, indexlst2, labellst2 = [],[],[],[]
    model_name = 'SARSMERSS1_checkpoint-600_SARSMERSesm2'
    for file in os.listdir(path):
        if file.endswith(label_file1):
            df = pd.read_csv (path + file, index_col = 'seqID')
            df_SM = df[df['species'] !='SARS2']
            indexlst1 = df_SM.index.tolist()
            labellst1 = df_SM['label'].tolist()
            
            df_S2 = df[df['species'] =='SARS2']
            indexlst2 = df_S2.index.tolist()
            labellst2 = df_S2['label'].tolist()
            
    print (len(labellst1), len(labellst2))        
    # print (indexlst2)
    
    
    df_array1 = pd.DataFrame()
    for file in os.listdir(path):
        
        if file.endswith ('checkpoint-600_SARSMERSembedding_SARSMERSesm2.pkl'):
            with open(path + file, 'rb') as f:
                data1 = pickle.load(f)
                array1 = np.array(data1)[:,:245, :]
                print(array1.shape)
                sample_num1 = array1.shape[0]
                reshaped_array1 = array1.reshape(sample_num1,-1)
                # print (reshaped_array1.shape)
                df_array1 = pd.DataFrame (data = reshaped_array1, index = indexlst1)
                df_array1['label'] = labellst1
                # print (df_array1.shape)
    
    
    label_file2 = 'df_metadata_forS1_SARS2_testing.csv'
    indexlst_test, labellst_test, variantlst_test = [], [], []
    for file in os.listdir(path):
        if file.endswith(label_file2):
            df = pd.read_csv (path + file, index_col = 'seqID')
            
            indexlst_test = df.index.tolist()
            variantlst_test = df.Variant.tolist()
    
    # print (len(labellst1), len(labellst2))
    # print (indexlst_test[:4])
    
    df_array2Alpha2 = pd.DataFrame()
    
    for file in os.listdir(path):
        
        if file.endswith ('checkpoint-600_SARS2S1embedding_SARSMERSesm2.pkl'):
            with open(path + file, 'rb') as f:
                data2 = pickle.load(f)
                array2 = np.array(data2)[:,:245, :]
                sample_num2 = array2.shape[0]
                reshaped_array2 = array2.reshape(sample_num2,-1)
                # print (reshaped_array2.shape)
                df_array2 = pd.DataFrame (data = reshaped_array2)
                df_array2['Variant'] = variantlst_test
                df_array2Alpha = df_array2[df_array2['Variant'] == 'Alpha']
                df_array2Alpha2 = df_array2Alpha.iloc[:, :-1]
                df_array2Alpha2['label'] = labellst2
                print (df_array2Alpha2.shape)
                
    df_all = pd.concat([df_array1, df_array2Alpha2])
    # df_all = shuffle(df_all)
    X = np.array(df_all.iloc[:, :-1]).reshape((df_all.shape[0], 560, 560))
    print (X.shape)
    ylst = df_all.iloc[:,-1].tolist()
    ylst = [int(i) for i in ylst]
    y = np.array(ylst)
    print (y.shape)
    
    return X, y, model_name

# loading_data()
# print(loading_data()[1])
