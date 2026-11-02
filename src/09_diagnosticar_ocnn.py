from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

PROYECTO = Path(__file__).resolve().parents[1]
DATOS = PROYECTO / "data" / "processed"
RESULTADOS = PROYECTO / "results"

modelo = np.load(RESULTADOS / "ocnn_pesos_y_umbral.npz")
W = modelo["W"]
V = modelo["V"]
umbral = float(modelo["umbral"])

print(f"Umbral: {umbral:.12g}")
print(f"Norma de W: {np.linalg.norm(W):.12g}")
print(f"Norma de V: {np.linalg.norm(V):.12g}")

archivos = [
    ("Entrenamiento", DATOS / "train.csv"),
    *[
        (archivo.stem, archivo)
        for archivo in sorted(DATOS.glob("test_set_*.csv"))
    ],
]

for nombre, archivo in archivos:
    datos = pd.read_csv(archivo)

    if "class" in datos.columns:
        y_real = datos["class"].astype(int).to_numpy()
        X = datos.drop(columns=["class"]).to_numpy(dtype=np.float32)
    else:
        y_real = None
        X = datos.to_numpy(dtype=np.float32)

    puntuaciones = (
        np.maximum(0, X @ W) @ V
    ).ravel() - umbral

    pred = np.where(puntuaciones >= 0, 1, -1)

    print(f"\n{nombre}")
    print(
        "Puntuaciones: "
        f"mín={puntuaciones.min():.12g}, "
        f"máx={puntuaciones.max():.12g}, "
        f"desv.={puntuaciones.std():.12g}"
    )
    print(f"Predicciones: {pd.Series(pred).value_counts().to_dict()}")

    if y_real is not None:
        print("Matriz de confusión (filas reales; columnas predichas, orden -1, 1):")
        print(confusion_matrix(y_real, pred, labels=[-1, 1]))