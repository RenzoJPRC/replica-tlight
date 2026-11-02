from pathlib import Path
import pandas as pd

PROYECTO = Path(__file__).resolve().parents[1]
RESULTADOS = PROYECTO / "results"

archivos = {
    "OCSVM": "ocsvm_metricas.csv",
    "OCNN (ReLU)": "ocnn_metricas.csv",
    "Isolation Forest": "if_metricas.csv",
}

tablas = []

for modelo, nombre_archivo in archivos.items():
    archivo = RESULTADOS / nombre_archivo

    if not archivo.exists():
        raise FileNotFoundError(f"No se encontró: {archivo}")

    datos = pd.read_csv(archivo)
    datos = datos[datos["conjunto"].str.startswith("test_set_")].copy()
    datos.insert(0, "modelo", modelo)
    tablas.append(datos)

comparacion = pd.concat(tablas, ignore_index=True)

orden_conjuntos = [f"test_set_{i}" for i in range(1, 6)]
comparacion["conjunto"] = pd.Categorical(
    comparacion["conjunto"],
    categories=orden_conjuntos,
    ordered=True,
)
comparacion = comparacion.sort_values(["conjunto", "modelo"])

columnas = [
    "modelo",
    "conjunto",
    "accuracy",
    "precision_ponderada",
    "recall_ponderado",
    "f1_ponderado",
    "precision_anomalias",
    "recall_anomalias",
    "f1_anomalias",
]

comparacion[columnas].to_csv(
    RESULTADOS / "comparacion_modelos.csv",
    index=False,
)

print(comparacion[columnas].round(4).to_string(index=False))
print(f"\nComparación guardada en: {RESULTADOS / 'comparacion_modelos.csv'}")