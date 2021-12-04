import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

def traer_datasets():
    df_1 = pd.read_csv(
        'https://docs.google.com/spreadsheets/d/1gvZ03uAL6THwd04Y98GtIj6SeAHiKyQY5UisuuyFSUs/export?format=csv', low_memory=False
    )
    df_2 = pd.read_csv(
        'https://docs.google.com/spreadsheets/d/1wduqo5WyYmCpaGnE81sLNGU0VSodIekMfpmEwU0fGqs/export?format=csv', low_memory=False
    )
    pd.options.display.max_columns = None

    df = df_2.merge(df_1, left_on = 'id', right_on = 'id')
    df.sort_values(by=['id'], inplace=True, ascending=True)
    df = df.dropna(subset=['llovieron_hamburguesas_al_dia_siguiente'])

    
    df_sin_target = df.copy().drop('llovieron_hamburguesas_al_dia_siguiente', 1)
    solo_target = df.copy()['llovieron_hamburguesas_al_dia_siguiente'].to_frame()
    
    return df, df_sin_target, solo_target


def separar_dataset(x, y):
    X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.1, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test

def aplicar_dummy_variables_encoding(df, columnas):
    df_encodeado = pd.get_dummies(df, columns=columnas, dummy_na=False, drop_first=True)
    return df_encodeado

def aplicar_ordinal_encoding(df, columnas):
    oe = OrdinalEncoder(dtype='int')
    return oe.fit_transform(df[columnas])
    
def feature_engineering_general(df_train, df_test):
    X_train = df_train.copy(deep=True)
    X_test = df_test.copy(deep=True)
    
    X_train.fillna(np.nan, inplace = True)
    X_test.fillna(np.nan, inplace = True)
    media_temp_max = X_train['temp_max'].mean()
    media_temp_min = X_train['temp_min'].mean()
    media_temp_temprano = X_train['temperatura_temprano'].mean()
    media_vel_viento_temprano = X_train['velocidad_viendo_temprano'].mean()
    X_train['temp_max'].replace(np.nan, media_temp_max , inplace = True)
    X_train['temp_min'].replace(np.nan, media_temp_min, inplace = True)
    X_train['temperatura_temprano'].replace(np.nan, media_temp_temprano , inplace = True)
    X_train['velocidad_viendo_temprano'].replace(np.nan, media_vel_viento_temprano , inplace = True)
    
    X_train['presion_atmosferica_tarde'].replace('.+\..+\..+', np.nan, inplace=True, regex=True)
    X_train.astype({'presion_atmosferica_tarde': 'float64'}).dtypes
    #X_train['presion_atmosferica_tarde'] = pd.to_numeric(X_train['presion_atmosferica_tarde'])
    
    eliminar_features_categoricas_general(X_train)
    X_train.reset_index()
    X_train = aplicar_dummy_variables_encoding(X_train,['llovieron_hamburguesas_hoy'])
    imputar_missings_KNN(X_train)
    
    X_test['temp_max'].replace(np.nan, media_temp_max , inplace = True)
    X_test['temp_min'].replace(np.nan, media_temp_min, inplace = True)
    X_test['temperatura_temprano'].replace(np.nan, media_temp_temprano , inplace = True)
    X_test['velocidad_viendo_temprano'].replace(np.nan, media_vel_viento_temprano, inplace = True)
    
    X_test['presion_atmosferica_tarde'].replace('.+\..+\..+', np.nan, inplace=True, regex=True)
    X_test.astype({'presion_atmosferica_tarde': 'float64'}).dtypes
    #X_test['presion_atmosferica_tarde'] = pd.to_numeric(X_test['presion_atmosferica_tarde'])
    
    eliminar_features_categoricas_general(X_test)
    X_test.reset_index()
    X_test = aplicar_dummy_variables_encoding(X_test, ['llovieron_hamburguesas_hoy'])
    imputar_missings_KNN(X_test)
    return X_train, X_test
    
    
def eliminar_features_categoricas_general(df):
    df.drop(['id','dia','barrio', 'direccion_viento_tarde', 'direccion_viento_temprano', 'rafaga_viento_max_direccion'], axis=1, inplace=True)

def imputar_missings_KNN(df):
    imputer = KNNImputer(n_neighbors = 3, weights="uniform")
    imputer.fit_transform(df)
    
def categorizar_humedad(humedad):
    if humedad >= 79:
        return "alta"
    elif humedad <= 30:
        return "baja"
    return "media"