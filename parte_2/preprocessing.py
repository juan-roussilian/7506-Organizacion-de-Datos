import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler

def aplicar_dummy_variables_encoding(df, columnas):
    df_encodeado = pd.get_dummies(df, columns=columnas, dummy_na=True, drop_first=True)
    return df_encodeado

def aplicar_ordinal_encoding(df, columnas):
    oe = OrdinalEncoder(dtype='int')
    return oe.fit_transform(df[columnas])

def feature_engineering_basico(df_a_preprocesar, features_eliminables, dummy_llovieron_hamburguesas_hoy):
    df = df_a_preprocesar.copy(deep=True)
    df.fillna(np.nan, inplace = True)
    
    features_continuas = ['id','horas_de_sol','humedad_tarde', 'humedad_temprano', 'mm_evaporados_agua', 'mm_lluvia_dia', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_tarde', 'presion_atmosferica_temprano', 'rafaga_viento_max_velocidad','temp_max', 'temp_min', 'temperatura_tarde', 'temperatura_temprano',  'velocidad_viendo_tarde','velocidad_viendo_temprano'] 
    
    if(dummy_llovieron_hamburguesas_hoy):
        df = aplicar_dummy_variables_encoding(df, ['llovieron_hamburguesas_hoy'])
        features_continuas.append('llovieron_hamburguesas_hoy_si')
        features_continuas.append('llovieron_hamburguesas_hoy_nan')
    
    df['presion_atmosferica_tarde'].replace('.+\..+\..+', np.nan, inplace=True, regex=True)
    df.astype({'presion_atmosferica_tarde': 'float64'}).dtypes
    eliminar_features(df, features_eliminables)
        
    df = imputar_missings_iterative(df, features_continuas)
    df.reset_index()
    df.sort_values(by=['id'], inplace=True, ascending=True)
    return df

def preprocesamiento_GNB(dataframes):
    dataframes_procesados = []
    for df in dataframes:
        df_procesado = feature_engineering_basico(
            df, 
            ['dia','barrio', 'direccion_viento_tarde', 'direccion_viento_temprano', 'rafaga_viento_max_direccion', 'llovieron_hamburguesas_hoy'], 
            dummy_llovieron_hamburguesas_hoy=False
        )
        dataframes_procesados.append(df_procesado)
    return dataframes_procesados
    
    
def preprocesamiento_basico(dataframes):
    dataframes_procesados = []
    for df in dataframes:
        df_procesado = feature_engineering_basico(
            df, 
            ['dia','barrio', 'direccion_viento_tarde', 'direccion_viento_temprano', 'rafaga_viento_max_direccion'],
            dummy_llovieron_hamburguesas_hoy=True
        )
        dataframes_procesados.append(df_procesado)
    return dataframes_procesados

def normalizar_datos_standard(df_train):
    df_a_normalizar = df_train.copy()
   
    scaler = StandardScaler()
    scaler.fit(df_a_normalizar)
    return scaler

def normalizar_datos_minmax(df_train):
    
    df_a_normalizar = df_train.copy()

    scaler = MinMaxScaler()
    scaler.fit(df_a_normalizar)
    return scaler

    
def reduccion_TSNE(df):
    return TSNE(n_components=3).fit_transform(df)

def reduccion_MDS(df):
    return MDS(n_components=6).fit_transform(df)
    
def eliminar_features(df, columnas):
    df.drop(columnas, axis=1, inplace=True)

def imputar_missings_KNN(df):
    imputer = KNNImputer(n_neighbors = 3, weights="uniform")
    imputer.fit_transform(df)
    
def imputar_missings_iterative(df, columnas_continuas):
    imputer = IterativeImputer()
    array_imputeado = imputer.fit_transform(df[columnas_continuas])
    df_imputeado = pd.DataFrame(array_imputeado, columns=columnas_continuas)
    df_imputeado.set_index('id', inplace=True)
    df_imputeado = df_imputeado.sort_values('id')
    return df_imputeado