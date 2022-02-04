import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

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
    solo_target = df[['id','llovieron_hamburguesas_al_dia_siguiente']].copy()
    
    return df, df_sin_target, solo_target

def traer_dataset_prediccion_final():
    df = pd.read_csv('https://docs.google.com/spreadsheets/d/1mR_JNN0-ceiB5qV42Ff9hznz0HtWaoPF3B9zNGoNPY8/export?format=csv', low_memory=False)
    df.sort_values(by=['id'], inplace=True, ascending=True)
    return df

def separar_dataset(x, y, test_size=0.1):
    X_train, X_test, y_train, y_test= train_test_split(x, y, test_size, random_state=0, stratify=y['llovieron_hamburguesas_al_dia_siguiente'])
    return X_train, X_test, y_train, y_test

def separar_dataset_train_val_holdout(x, y):
    X_train, X_test, y_train, y_test = separar_dataset(x, y, test_size=0.3)
    X_test, X_holdout, y_test, y_holdout = separar_dataset(X_test, y_test, test_size=0.333)
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout

def encontrar_hiperparametros_RGSCV(clf, params, x_np, y_np):
    rgscv = RandomizedSearchCV(clf, params, n_iter=100, scoring='roc_auc', n_jobs=-2, return_train_score=True).fit(x_np, y_np)
    return rgscv.best_params_

def encontrar_hiperparametros_GSCV(clf, params, x_np, y_np):
    gsvc = GridSearchCV(clf, params, scoring='roc_auc', n_jobs=-2, cv=5).fit(x_np, y_np)    
    return gsvc.best_params_

def mapear_target_binario(x):
    if(x == 'si'):
        return 1
    return 0

def mapear_target_binario_a_categorico(x):
    if(x == 1):
        return 'si'
    return 'no'

def exportar_prediccion_final(ids, predicciones, nombre_modelo):
    with open("predicciones/"+nombre_modelo+".csv", "w") as archivo:
        archivo.write("id,tiene_alto_valor_adquisitivo\n")       
        for medicion in range(len(ids)):
            archivo.write(str(ids[medicion]) + "," + str(predicciones[medicion]) + "\n")
