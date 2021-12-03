import pandas as pd
from sklearn.model_selection import train_test_split

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
    
    df_sin_target = df.copy().drop('llovieron_hamburguesas_al_dia_siguiente', 1)
    solo_target = df.copy()['llovieron_hamburguesas_al_dia_siguiente'].to_frame()
    
    return df, df_sin_target, solo_target


def separar_dataset(x, y):
    return train_test_split(x, y, test_size=0.1, random_state=0, stratify=y)

def aplicar_one_hot_encoding(df, columnas):
    return pd.get_dummies(df_con_encoding, columnas, dummy_na=True, drop_first=True)

def aplicar_ordinal_encoding(df, columnas):
    oe = oe = OrdinalEncoder(dtype='int')
    return oe.fit_transform(df[columnas])
    
def feature_engineering_general(df):
    df['presion_atmosferica_tarde'].replace('.+\..+\..+', np.nan, inplace=True, regex=True)
    df['presion_atmosferica_tarde'] = pd.to_numeric(df['presion_atmosferica_tarde'])
