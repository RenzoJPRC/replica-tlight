from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

PROYECTO = Path(__file__).resolve().parents[1]
DATOS = PROYECTO / "data" / "processed"
RESULTADOS = PROYECTO / "results"

modelo = joblib.load(RESULTADOS / "ocsvm.joblib")

print("Parámetros del modelo:")
print(modelo.get_params())
print(f"Variables usadas: {modelo.n_features_in_}")
print(f"Vectores de soporte: {modelo.support_vectors_.shape[0]}")

for archivo in sorted(DATOS.glob("test_set_*.csv")):
    datos = pd.read_csv(archivo)
    y_real = datos["class"].astype(int)
    X = datos.drop(columns=["class"])

    pred = modelo.predict(X)
    puntuacion = modelo.decision_function(X)

    matriz = confusion_matrix(y_real, pred, labels=[-1, 1])

    print("\n" + "=" * 60)
    print(archivo.name)
    print("Matriz de confusión (filas reales, columnas predichas)")
    print("Orden de clases: -1 = anomalía, 1 = normal")
    print(matriz)
    print("Predicciones:")
    print(pd.Series(pred).value_counts().sort_index().to_dict())
    print("Puntuación media según clase real:")
    print(pd.DataFrame({
        "clase_real": y_real,
        "puntuacion": puntuacion,
    }).groupby("clase_real")["puntuacion"].mean().to_dict())