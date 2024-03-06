# Импорт библиотек

import pandas as pd
import numpy as np

import faiss

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

seed = 4

# Функция чтения данных

def read_data(path):
    return pd.read_csv(path, index_col=0)

# Функция удаления столбцов

def remove_col(df, col):
    return df.drop(col, axis=1)

def scale_df(scaler, df):
    return scaler.transform(df)

def faiss_index_builder(df_scaled, n_clusters, nprobe):
    
    dim = df_scaled.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    
    index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
    
    index.train(np.ascontiguousarray(pd.DataFrame(df_scaled).sample(frac=0.16, random_state=seed)).astype('float32'))
    index.add(np.ascontiguousarray(df_scaled).astype('float32'))
    
    index.nprobe = nprobe
    
    return index

def accuracy_five(targets, idx, base_index):
    acc = sum(int(targets in [main_df.index[r] for r in el]) for targets, el in zip(targets.values.tolist(), idx.tolist()))
    return 100 * acc / len(idx)

# Чтение данных

base_path = '...'
validation_path = '...'
validation_answers_path = '...'

main_df = read_data(base_path)
valid_df = read_data(validation_path)
valid_answ_df = read_data(validation_answers_path)

# Настройка столбцов

adjective_cols = ['6', '21', '25', '33', '44', '70', '59', '65']

main_df = remove_col(main_df, adjective_cols)
valid_df = remove_col(valid_df, adjective_cols)
targets_validation = valid_answ_df['Expected']

# Маштабизация данных
scaler = RobustScaler()
scaler.fit(main_df)

main_scaled = scale_df(scaler, main_df)
valid_scaled = scale_df(scaler, valid_df)

n_clusters = 101
nprobe = 3

base_index = {k: v for k, v in enumerate(main_df.index.to_list())}

faiss_index = faiss_index_builder(main_scaled, n_clusters, nprobe)


_, idx = faiss_index.search(np.ascontiguousarray(valid_scaled).astype('float32'), 5)

print(accuracy_five(targets_validation, idx, base_index))
