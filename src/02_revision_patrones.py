from pathlib import Path
import pandas as pd

PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATOS = PROYECTO / "data"

archivos = [
    CARPETA_DATOS / "train_data" / "train_dataset.csv",
    *sorted((CARPETA_DATOS / "test_data").glob("test_set_*.csv")),
]

for archivo in archivos:
    datos = pd.read_csv(archivo)

    columna_clase = next(
        (c for c in datos.columns if c.lower() in {"class", "label", "target", "y"}),
        None,
    )

    variables = [
        c for c in datos.columns
        if c != columna_clase and not c.lower().startswith("unnamed")
    ]

    patrones_unicos = datos[variables].drop_duplicates().shape[0]

    print("\n" + "=" * 65)
    print(archivo.name)
    print(f"Registros: {len(datos):,}")
    print(f"Patrones distintos de señales: {patrones_unicos:,}")
    print(f"Registros que repiten un patrón: {len(datos) - patrones_unicos:,}")

    if columna_clase:
        etiquetas_por_patron = (
            datos.groupby(variables, dropna=False, sort=False)[columna_clase]
            .nunique()
        )
        conflictos = int((etiquetas_por_patron > 1).sum())
        print(f"Patrones con etiquetas normal y anómala: {conflictos:,}")