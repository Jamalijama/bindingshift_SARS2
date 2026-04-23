import pandas as pd
import os
from sklearn.utils import shuffle
import string
import random
import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
import Translator
import pickle
import numpy as np

 # 1. 从微调后的模型提取序列Embedding
def get_sequence_embedding(sequence, model, tokenizer):
    """输入单个蛋白序列，返回其Embedding（使用<cls> token的输出）"""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            sequence,
            #padding="max_length",
            truncation=True,
            #max_length=max_length,
            return_tensors="pt"
        ).to(device)  # 移动到模型所在设备（GPU/CPU）

        outputs = model(**inputs, output_hidden_states=True)
        # 取最后一层的<cls> token输出作为序列Embedding
        cls_embedding = outputs.hidden_states[-1][:, :, :].squeeze()
        cls_embedding = cls_embedding[1:-1]
    return cls_embedding.cpu().numpy()  # 转为numpy数组


# extract_embedding


df_test = pd.read_csv ('./data/df_parsed_SpikeS1_MERS_SARS.csv', index_col = 'seqID', encoding = 'gbk')

cdslst = df_test['SpikeS1'].tolist()
model_path = "./esm2_finetuned_SARS2S1/" 

for m in os.listdir(model_path)[40:]:
    model_name = model_path + m
    print (model_name)
        
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    len_cds = 256
    pkl_path = './embedding/embedding_SARSMERS_esm2SARS2/SARS2S1esm2_' + m + '_SARSMERSembedding.pkl'
    print (pkl_path)
    embed_lst = []
    for i in range(df_test.shape[0]):
        
        if i%500 ==0:
            print(i)
        
        test_sequence = cdslst[i]
        embedding = get_sequence_embedding(
            test_sequence,
            model=model,
            tokenizer=tokenizer
        )
        embedding = np.concatenate((embedding, np.zeros((len_cds - embedding.shape[0], 1280))))
        #print (embedding.shape) 
        embed_lst.append(embedding)
        
    with open(pkl_path,'wb') as f_pkl:
        pickle.dump(embed_lst,f_pkl) 
        
    print('embedding_finished!!!!!!')     
    print(f"测试序列: {embedding[-1]}...（长度100）")
    print(f"序列Embedding形状: {embedding.shape}")  # 输出应为(384,)（对应8M模型）
    
    


# for m in os.listdir(model_path):
#     if m.startswith ('checkpoint-'):
#         model_name = model_path + m
#         print (m)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForMaskedLM.from_pretrained(model_name)
#         cds_lst = ['pb2', 'pb1', 'pa', 'np']
#         len_lst = [768,768,768,512, 256]
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model = model.to(device)
        
#         len_cds = len_lst[-1]
#         pkl_path = './data_SARS2/' + m + '_spikeprot_S1dedup_S1high95_Shigh95_132204_v2_testing32204_embedding.pkl'
        
#         embed_lst = []
#         for i in range(len(df_test)):
#             print(i)
#             test_sequence = str(df_test.loc[i,'prS1'])
#             embedding = get_sequence_embedding(
#                 test_sequence,
#                 model=model,
#                 tokenizer=tokenizer
#             )
#             embedding = np.concatenate((embedding, np.zeros((len_cds - embedding.shape[0], 1280))))
#             #print (embedding.shape) 
#             embed_lst.append(embedding)
#         with open(pkl_path,'wb') as f_pkl:
#             pickle.dump(embed_lst,f_pkl)   
#         print('embedding_finished!!!!!!')     
#         print(f"测试序列: {embedding[-1]}...（长度100）")
#         print(f"序列Embedding形状: {embedding.shape}")  # 输出应为(384,)（对应8M模型）
