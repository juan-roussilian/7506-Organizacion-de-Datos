import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV

def encontrar_hiperparametros_RGSCV(clf, params, X, y):
    x_np = X.to_numpy()
    y_np = y.to_numpy()
    rgscv = RandomizedSearchCV(clf, params, n_iter=100, scoring='roc_auc', n_jobs=-1, return_train_score=True).fit(x_np, y_np)
    return rgscv.best_params_

def mapear_target_binario(x):
    if(x == 'si'):
        return 1
    return 0
