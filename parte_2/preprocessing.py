import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction import FeatureHasher


def aplicar_dummy_variables_encoding(df, columnas):
    df_encodeado = pd.get_dummies(df, columns=columnas, dummy_na=True, drop_first=True)
    return df_encodeado
    
def aplicar_hashing_trick(df, columnas, nuevas_features, reset_index=True):
    df_hasheado = df.copy(deep=True)
    
    if reset_index:
        df_hasheado.reset_index(inplace=True)
        
    for columna, n_nuevas_features in zip(columnas, nuevas_features):
        hasher = FeatureHasher(n_features=n_nuevas_features, input_type='string')
        features_hashadas = hasher.fit_transform(df_hasheado[columna].astype(str)).todense()
        features_hashadas = pd.DataFrame(features_hashadas).add_prefix(columna+'_')
        df_hasheado = df_hasheado.join(features_hashadas)
        df_hasheado = df_hasheado.drop(columna, axis=1)

    return df_hasheado

def entrenar_normalizador_standard(df_train):
    df_a_normalizar = df_train.copy()
   
    scaler = StandardScaler()
    scaler.fit(df_a_normalizar)
    return scaler

def entrenar_normalizador_minmax(df_train):
    df_a_normalizar = df_train.copy()

    scaler = MinMaxScaler()
    scaler.fit(df_a_normalizar)
    return scaler

def normalizar_dataframe(df, scaler):
    df_normalizado = scaler.transform(df.copy())
    return df_normalizado

def limpiar_datos(df_a_limpiar):
    df = df_a_limpiar.copy(deep=True)
    df.fillna(np.nan, inplace = True)
    
    df['presion_atmosferica_tarde'].replace('.+\..+\..+', np.nan, inplace=True, regex=True)
    df.astype({'presion_atmosferica_tarde': 'float64'}).dtypes
    return df

def reduccion_TSNE(df):
    return TSNE(n_components=3).fit_transform(df)

def reduccion_MDS(df):
    return MDS(n_components=6).fit_transform(df)

def reduccion_PCA(df, dim_destino):
    return PCA(n_components=dim_destino).fit_transform(df)
    
def eliminar_features(df, columnas):
    df.drop(columnas, axis=1, inplace=True)
    df.reset_index()

def imputar_missings_KNN(df):
    imputer = KNNImputer(n_neighbors = 3, weights="uniform")
    imputer.fit_transform(df)
    
def entrenar_iterative_imputer(df_train):
    imputer = IterativeImputer()
    imputer.fit(df_train)
    return imputer
    
def imputar_missings_iterative(df, imputer_entrenado):
    array_imputeado = imputer_entrenado.transform(df)
    df_imputeado = pd.DataFrame(array_imputeado, columns=df.columns)
    df_imputeado.set_index('id', inplace=True)
    df_imputeado = df_imputeado.sort_values('id') 
    return df_imputeado