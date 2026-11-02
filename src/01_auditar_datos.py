from pathlib import Path
import pandas as pd

# Ubica la carpeta principal del proyecto
PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATOS = PROYECTO / "data"

archivo_entrenamiento = (
    CARPETA_DATOS / "train_data" / "train_dataset.csv"
)
archivos_prueba = sorted(
    (CARPETA_DATOS / "test_data").glob("test_set_*.csv")
)

archivos = [archivo_entrenamiento, *archivos_prueba]

for archivo in archivos:
    print("\n" + "=" * 70)
    print(f"Archivo: {archivo.relative_to(PROYECTO)}")

    if not archivo.exists():
        print("ERROR: no se encontró este archivo.")
        continue

    datos = pd.read_csv(archivo)

    print(f"Filas: {datos.shape[0]:,}")
    print(f"Columnas: {datos.shape[1]}")
    print(f"Valores vacíos: {datos.isna().sum().sum():,}")
    print(f"Filas duplicadas: {datos.duplicated().sum():,}")

    columnas_etiqueta = [
        columna
        for columna in datos.columns
        if columna.lower() in {"class", "label", "target", "y"}
    ]

    if columnas_etiqueta:
        columna = columnas_etiqueta[0]
        print(f"Conteo de etiquetas en '{columna}':")
        print(datos[columna].value_counts(dropna=False).to_string())
    else:
        print("No se encontró una columna de etiquetas.")

    columnas_indice = [
        columna for columna in datos.columns
        if columna.lower().startswith("unnamed")
    ]
    print(f"Posibles columnas de índice: {columnas_indice}")
    print(f"Columnas: {list(datos.columns)}")