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
    
    label_file2 = 'df_metadata_forS1_SARS2_testing.csv'
    model_name = 'SARS2S1_checkpoint-600_SARSMERSesm2'   
    
    indexlst_test, labellst_test, variantlst_test = [], [], []
    for file in os.listdir(path):
        if file.endswith(label_file2):
            df = pd.read_csv (path + file, index_col = 'seqID')
            
            indexlst_test = df.index.tolist()
            # variantlst_test = df.Variant.tolist()
            labellst_test = df.label.tolist()
            print (df.label.value_counts())
            min_num = min(df.label.value_counts())
            # print (min_num)
    
    
    
    for file in os.listdir(path):
        
        if file.endswith ('checkpoint-600_SARS2S1embedding_SARSMERSesm2.pkl'):
            with open(path + file, 'rb') as f:
                data2 = pickle.load(f)
                array2 = np.array(data2)[:,:245, :]
                sample_num2 = array2.shape[0]
                reshaped_array2 = array2.reshape(sample_num2,-1)
                print (reshaped_array2.shape)
                df_array2 = pd.DataFrame (data = reshaped_array2, index = indexlst_test)
                df_array2['label'] = labellst_test
                df = shuffle(df, random_state = 1)
                
                df_0 = df[df['label'] == 0]
                df_1 = df[df['label'] == 1].sample(n = min_num, random_state = 1)
                df_2 = df[df['label'] == 2].sample(n = min_num, random_state = 1)
                df_label_all = pd.concat([df_0, df_1, df_2])
                index_lst_selected = df_label_all.index.tolist()

                df_array = df_array2.loc[index_lst_selected]
                df_array = shuffle(df_array, random_state = 1)
                
                # df_array.to_csv(path + 'df_SARS2S1embedding_checkpoint-600_SARSMERSesm2.csv')

    X = np.array(df_array.iloc[:, :-1]).reshape((df_array.shape[0], 560, 560))
    print (X.shape)
    ylst = df_array.iloc[:,-1].tolist()
    ylst = [int(i) for i in ylst]
    y = np.array(ylst)
    print (y.shape)
    
    return X, y, model_name


# loading_data()


