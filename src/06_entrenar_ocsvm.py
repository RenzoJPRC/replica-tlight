from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

PROYECTO = Path(__file__).resolve().parents[1]
DATOS = PROYECTO / "data" / "processed"
RESULTADOS = PROYECTO / "results"
RESULTADOS.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATOS / "train.csv")
y_train = np.ones(len(X_train), dtype=int)

modelo = OneClassSVM(
    kernel="poly",
    degree=3,
    coef0=4,
    nu=0.1,
    gamma=0.1,
)

print("Entrenando OCSVM con 41.580 registros...")
modelo.fit(X_train)
joblib.dump(modelo, RESULTADOS / "ocsvm.joblib")

def calcular_metricas(y_real, y_pred):
    precision_p, recall_p, f1_p, _ = precision_recall_fscore_support(
        y_real, y_pred, average="weighted", zero_division=0
    )

    resultado = {
        "accuracy": accuracy_score(y_real, y_pred),
        "precision_ponderada": precision_p,
        "recall_ponderado": recall_p,
        "f1_ponderado": f1_p,
    }

    if len(np.unique(y_real)) > 1:
        precision_a, recall_a, f1_a, _ = precision_recall_fscore_support(
            y_real, y_pred, average="binary", pos_label=-1, zero_division=0
        )
        resultado.update({
            "precision_anomalias": precision_a,
            "recall_anomalias": recall_a,
            "f1_anomalias": f1_a,
        })

    return resultado

filas = []

pred_train = modelo.predict(X_train)
filas.append({
    "conjunto": "Entrenamiento",
    **calcular_metricas(y_train, pred_train),
})

for archivo in sorted(DATOS.glob("test_set_*.csv")):
    datos = pd.read_csv(archivo)
    y_real = datos["class"].astype(int)
    X_test = datos.drop(columns=["class"])

    pred = modelo.predict(X_test)
    puntuaciones = modelo.decision_function(X_test)

    filas.append({
        "conjunto": archivo.stem,
        **calcular_metricas(y_real, pred),
    })

    pd.DataFrame({
        "class_real": y_real,
        "class_predicha": pred,
        "decision_score": puntuaciones,
    }).to_csv(
        RESULTADOS / f"ocsvm_{archivo.stem}_predicciones.csv",
        index=False,
    )

tabla = pd.DataFrame(filas)
tabla.to_csv(RESULTADOS / "ocsvm_metricas.csv", index=False)

print("\nResultados de OCSVM:")
print(tabla.round(4).to_string(index=False))
print(f"\nModelo y resultados guardados en: {RESULTADOS}")