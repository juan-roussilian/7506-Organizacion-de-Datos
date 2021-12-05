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
    df_encodeado = pd.get_dummies(df, columns=columnas, dummy_na=True, drop_first=True)
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
    
    features_poco_influyentes = ['dia','barrio', 'direccion_viento_tarde', 'direccion_viento_temprano', 'rafaga_viento_max_direccion']
    features_continuas = ['id','horas_de_sol','humedad_tarde', 'humedad_temprano', 'mm_evaporados_agua', 'mm_lluvia_dia', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_tarde', 'presion_atmosferica_temprano', 'rafaga_viento_max_velocidad','temp_max', 'temp_min', 'temperatura_tarde', 'temperatura_temprano',  'velocidad_viendo_tarde','velocidad_viendo_temprano', 'llovieron_hamburguesas_hoy_si','llovieron_hamburguesas_hoy_nan']
    
    X_train['temp_max'].replace(np.nan, media_temp_max , inplace = True)
    X_train['temp_min'].replace(np.nan, media_temp_min, inplace = True)
    X_train['temperatura_temprano'].replace(np.nan, media_temp_temprano , inplace = True)
    X_train['velocidad_viendo_temprano'].replace(np.nan, media_vel_viento_temprano , inplace = True)
    X_train['presion_atmosferica_tarde'].replace('.+\..+\..+', np.nan, inplace=True, regex=True)
    X_train.astype({'presion_atmosferica_tarde': 'float64'}).dtypes
    eliminar_features(X_train, features_poco_influyentes)
    X_train = aplicar_dummy_variables_encoding(X_train,['llovieron_hamburguesas_hoy'])
    X_train = imputar_missings_iterative(X_train, features_continuas)
    X_train.reset_index()
    
    
    X_test['temp_max'].replace(np.nan, media_temp_max , inplace = True)
    X_test['temp_min'].replace(np.nan, media_temp_min, inplace = True)
    X_test['temperatura_temprano'].replace(np.nan, media_temp_temprano , inplace = True)
    X_test['velocidad_viendo_temprano'].replace(np.nan, media_vel_viento_temprano , inplace = True)
    X_test['presion_atmosferica_tarde'].replace('.+\..+\..+', np.nan, inplace=True, regex=True)
    X_test.astype({'presion_atmosferica_tarde': 'float64'}).dtypes
    eliminar_features(X_test, features_poco_influyentes)
    X_test = aplicar_dummy_variables_encoding(X_test,['llovieron_hamburguesas_hoy'])
    X_test = imputar_missings_iterative(X_test, features_continuas)
    X_test.reset_index()
    
    return X_train, X_test
    
    
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
    
def categorizar_humedad(humedad):
    if humedad >= 79:
        return "alta"
    elif humedad <= 30:
        return "baja"
    return "media"