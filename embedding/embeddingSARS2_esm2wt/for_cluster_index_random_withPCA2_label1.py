from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, \
    normalized_mutual_info_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import pickle

# from imblearn.over_sampling import SMOTE

cnames = {
    'lightblue': '#ADD8E6',
    'deepskyblue': '#00BFFF',
    'cadetblue': '#5F9EA0',
    'cyan': '#00FFFF',
    'purple': '#800080',
    'orchid': '#DA70D6',
    'lightgreen': '#90EE90',
    'darkgreen': '#006400',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32',
    'deeppink': '#FF1493',
    'burlywood': '#DEB887',
    'red': '#FF0000',
    'indianred': '#CD5C5C',
    'darkred': '#8B0000',
}
color_num_list = list(range(1, 16, 1))

# print (len(color_num_list))
color_dict = dict(zip(color_num_list, cnames.values()))
# print (color_dict)
color_list0 = list(color_dict.values())
           

path_model = './'
model = path_model + 'esm2_t33_650M_UR50D_spikeprot_S1dedup_Shigh95_132204_v2_testing32204_embedding.pkl'

acc_1_lst = []
sh_score_1_lst = []
ch_score_1_lst = []
dbi_score_1_lst = []
ari_score_1_lst = []
nmi_score_1_lst = []
model_lst = []

df_labels = pd.read_csv ('../../data/df_metadata_forS1_testing.csv')
label_1 = df_labels['label1'].tolist()
print(len(label_1))
seq_idlst = df_labels['Accession ID'].tolist()
    
with open (model, 'rb') as f:
    data = pickle.load(f)
    array = np.array(data)    
    sample_num = array.shape[0]
    reshape_composition_array = array.reshape(sample_num,-1)
    print (array.shape)
    
    # model_lst.append(file_name)
            
    df_array0 = pd.DataFrame (data = reshape_composition_array, index = seq_idlst)
    df_array0['label1'] = label_1
    # df_array0['seqID'] = seq_idlst
    # df_array0 = df_array0[df_array0['label'].isin([0, 1, 2])]     

    file_name = 'esm2_t33_650M_UR50D_spikeprot_S1dedup'
    print (file_name)
        
    for rand_i in range(0,10,1):
        print (rand_i)
        
        model_lst.append(rand_i)
       
                              
        df_label0 = shuffle(df_labels, random_state = rand_i)
        # countslst = df_label['label'].value_counts().tolist()
        df_label_0 = df_label0[df_label0['label1'] == 0].sample(n = 600, random_state = rand_i)
        df_label_1 = df_label0[df_label0['label1'] == 1].sample(n = 600, random_state = rand_i)
        df_label_2 = df_label0[df_label0['label1'] == 2].sample(n = 600, random_state = rand_i)
        df_label_3 = df_label0[df_label0['label1'] == 3].sample(n = 600, random_state = rand_i)
        df_label_4 = df_label0[df_label0['label1'] == 4].sample(n = 600, random_state = rand_i)
                
        df_label = pd.concat([df_label_0, df_label_1, df_label_2]) #, df_array_3, df_array_4
        print (df_label['label1'].value_counts())
        
        index_lst = df_label['Accession ID'].tolist()
        
        df_array = df_array0.loc[index_lst]
       
        # df_array = df_array.sample(n = 3000, random_state = 1)
        reshape_composition_array = np.array(df_array.iloc[:,:-1])
        seq_idlst = index_lst
        
        print(reshape_composition_array.shape)
        label_1 = df_array.iloc[:,-1].tolist()
    
        y_types = list(set(label_1))
        y_num = len(y_types)
        print('label number: ', y_num)
    
        
        X_tsne = TSNE(learning_rate=100).fit_transform(reshape_composition_array)
        X_pca = PCA(n_components=2).fit_transform(reshape_composition_array)
        k_selected = len(y_types)
        print (k_selected)
        

        
        # df_pca = pd.DataFrame(X_pca, columns = ['PCA1', 'PCA2'])
        # df_tsne = pd.DataFrame(X_tsne, columns = ['tSNE1', 'tSNE2'])
        
        # df_tsne['seqID'] = seq_idlst
        # df_pca['seqID'] = seq_idlst
        
        # df_tsne.to_csv ('df_tsne_' + file_name + '_sampled_3000_' + str(rand_i) +'_label.csv')
        # df_pca.to_csv ('df_PCA_' + file_name + '_sampled_3000_' + str(rand_i) +'_label.csv')
        
        ac_cluster_1_pred = MiniBatchKMeans(n_clusters=k_selected, random_state=10).fit_predict(X_pca)
    
        # ac_cluster_1_pred = AgglomerativeClustering(n_clusters=2, linkage='complete').fit_predict(X_pca)
        # # 聚类效果评价
        # print('\nagglomerative clustering:')
        # # accuracy，值越大越好
        acc_1 = accuracy_score(label_1, ac_cluster_1_pred)
        # print('accuracy=',acc_1)
        # # 轮廓系数，值越大越好
        sh_score_1 = silhouette_score(reshape_composition_array, ac_cluster_1_pred, metric='euclidean')
        print('sh_score =', sh_score_1)
        # # acc_1Calinski-Harabaz Index，值越大越好
        ch_score_1 = calinski_harabasz_score(reshape_composition_array, ac_cluster_1_pred)
        print('ch_score =', ch_score_1)
        # #  Davies-Bouldin Index(戴维森堡丁指数)，值越小越好
        dbi_score_1 = davies_bouldin_score(reshape_composition_array, ac_cluster_1_pred)
        print('dbi_score =', dbi_score_1)
        # # 调整兰德指数，值越大越好
        ari_score_1 = adjusted_rand_score(label_1, ac_cluster_1_pred)
        print('ari_score =', ari_score_1)
        # # 标准化互信息，值越大越好
        nmi_score_1 = normalized_mutual_info_score(label_1, ac_cluster_1_pred)
        print('nmi_score =', nmi_score_1)
                
        acc_1_lst.append(acc_1)
        sh_score_1_lst.append(sh_score_1)
        ch_score_1_lst.append(ch_score_1)
        dbi_score_1_lst.append(dbi_score_1)
        ari_score_1_lst.append(ari_score_1)
        nmi_score_1_lst.append(nmi_score_1)
        
    del data
    del array
    del df_array0

    
df_evaluation = pd.DataFrame()
df_evaluation['accuracy_score'] = acc_1_lst
df_evaluation['silhouette_score'] = sh_score_1_lst
df_evaluation['calinski_harabasz_score'] = ch_score_1_lst
df_evaluation['davies_bouldin_score'] = dbi_score_1_lst
df_evaluation['adjusted_rand_score'] = ari_score_1_lst
df_evaluation['normalized_mutual_info_score'] = nmi_score_1_lst
df_evaluation['sampling'] = model_lst
df_evaluation.to_csv('./df_evaluate_cluster_' + file_name + '_withPCA_label1.csv')


