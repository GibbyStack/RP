'''
CURVA ROC (Receiver Operating Characteristic) - Característica Operativa del Receptor
Evaluar la calidad de los clasificadores multiclase

- El área AUC, es el area bajo la curva
- Herramienta estadística que se utiliza para medir el acierto en la predicción de 
  eventos binarios, es decir, eventos que bien ocurren o no ocurren
- Las curvas ROC suelen presentar una tasa de verdaderos positivos (TPR) en el eje Y 
  y una tasa de falsos positivos (FPR) en el eje X
- Esto significa que la esquina superior izquierda de la gráfica es el punto "ideal": 
  un FPR de cero y un TPR de uno.
- La "inclinación" de las curvas ROC también es importante, ya que es ideal para 
  maximizar la TPR y minimizar la FPR.

Las curvas ROC se utilizan normalmente en la clasificación binaria, donde TPR y FPR 
se pueden definir sin ambigüedades. En el caso de clasificación multiclase, 
se obtiene una noción de TPR o FPR solo después de binarizar la salida. Esto se puede
hacer de 2 maneras diferentes:

1. Esquema One-vs-Rest compara cada clase con todas las demás (asumidas como una);
2. Esquema One-vs-One compara cada combinación única de clases por pares.
'''

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np

# Método para generar la matriz de confución
def confusion_matrix(y_predicted, y_test, m):
    mc = np.zeros((m, m)) # Crear matriz de ceros
    for i in range(len(y_predicted)):
        mc[y_predicted[i]][y_test[i]] += 1 # Agregar clasificación predicha con respecto a la real
    return mc

# Método para obtener las estadisticas proporcionadas por la MC
def get_statistics_mc(mc, multiclass=True):
  if multiclass:
    FP = mc.sum(axis=1) - np.diag(mc) # False positives
    FN = mc.sum(axis=0) - np.diag(mc) # False negatives
    TP = np.diag(mc) # True positives
    TN = mc.sum() - (FP + FN + TP) # True negativas
    ACCr = sum(TP) / mc.sum() # Total accuracy
    TPR = TP / (TP + FN) # True Positive Rate (Recall, Sensitivity)
    TNR = TN / (TN + FP) # True Negative Rate (Specificity)
    PPV = TP / (TP + FP) # Positive Predictive Value (Precision)
    TPRa = TPR.mean()
    TNRa = TNR.mean()
    PPVa = PPV.mean() # Total precision
    return ACCr, PPVa, TPRa, TNRa
  else:
    TP = mc[0][0] # True positives
    FP = mc[0][1] # False positives
    TN = mc[1][1] # True negativas
    FN = mc[1][0] # False negatives
    ACC = (TP+TN) / (TP + FP + FN + TN) # Accuracy
    TPR = TP / (TP + FN) # True Positive Rate (Recall, Sensitivity)
    TNR = TN / (TN + FP) # True Negative Rate (Specificity)
    PPV = TP / (TP + FP) # Positive Predictive Value (Precision)
  return ACC, PPV, TPR, TNR
  # print(' MC '.center(15, '='))
  # print(mc)
  # print(''.center(15, '='))
  # print(f'FP = {FP}')
  # print(f'FN = {FN}')
  # print(f'TP = {TP}')
  # print(f'TN = {TN}')
  # print(f'ACC = {ACC}')
  # print(f'PPV = {PPV}')
  # print(f'TPR = {TPR}')
  # print(f'TNR = {TNR}')
  # print(f'ACC average = {ACCr}')
  # print(f'PPV average = {PPV.mean()}')
  # print(f'TPR average = {TPR.mean()}')
  # print(f'TNR average = {TNR.mean()}')

# Método para obtener la curva ROC
def ROC_curve(y_train, y_test, y_predicted, targets, pos_label, multiclass=True):
  label_binarizer = LabelBinarizer().fit(y_train)
  y_onehot_test = label_binarizer.transform(y_test)
  if multiclass: 
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(targets)):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, i],
            y_predicted[:, i],
            pos_label=pos_label,
            name=f"{targets[i]} vs el resto",
            color='C'+str(i),
            ax = ax
        )
    plt.title("One-vs-Rest multiclass ROC")
  else:
    RocCurveDisplay.from_predictions(
        y_test,
        y_predicted[:, pos_label],
        pos_label=pos_label,
        name=f"{targets[pos_label]} vs {targets[1-pos_label]}",
        color='C0'
    )
    plt.title("Curve ROC")
  plt.plot([0, 1], [0, 1], "k--", label="Curva ROC para (AUC = 0.5)")
  plt.axis("square")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.legend()
  plt.show()