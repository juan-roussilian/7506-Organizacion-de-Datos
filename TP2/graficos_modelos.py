import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, classification_report

def graficar_matriz_confusion(y_true, y_pred):
    nombres = sorted(set(y_true))
    matriz = confusion_matrix(y_true, y_pred, nombres)
    df_matriz = pd.DataFrame(matriz, nombres, nombres)

    plt.figure(dpi=100)
    sns.heatmap(df_matriz, annot=True, annot_kws={"size": 16}, fmt='g', square=True)
    plt.title("Matriz de confusion")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.show()

    
def graficar_auc_roc(y_test, y_pred):    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plot_roc(fpr, tpr, thresholds)
    print(f"AUC-ROC para este modelo es: {roc_auc_score(y_test, y_pred)}")
    
def mostrar_reporte_clasificacion(y_true, y_pred):
    print (classification_report(y_true, y_pred, labels=['no', 'si'] , digits=3))
    
def plot_roc(_fpr, _tpr, x):
    roc_auc = auc(_fpr, _tpr)

    plt.figure(figsize=(15, 10))
    plt.plot(
        _fpr, _tpr, color='darkorange', lw=2, label=f'AUC score: {roc_auc:.2f}'
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()