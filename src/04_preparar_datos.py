from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROYECTO = Path(__file__).resolve().parents[1]
DATOS = PROYECTO / "data"
SALIDA = DATOS / "processed"
SALIDA.mkdir(parents=True, exist_ok=True)

archivo_train = DATOS / "train_data" / "train_dataset.csv"
archivos_test = sorted((DATOS / "test_data").glob("test_set_*.csv"))

train = pd.read_csv(archivo_train)

# Se excluye solo el índice auxiliar; el entrenamiento no tiene etiqueta.
columnas = [
    c for c in train.columns
    if not c.lower().startswith("unnamed")
]
X_train = train[columnas].apply(pd.to_numeric, errors="raise")

if len(columnas) != 36:
    raise ValueError(f"Se esperaban 36 variables, se encontraron {len(columnas)}.")

escalador = MinMaxScaler()
X_train_escalado = escalador.fit_transform(X_train)

pd.DataFrame(
    X_train_escalado, columns=columnas
).to_csv(SALIDA / "train.csv", index=False)

resumen = [
    f"Entrenamiento: {len(X_train):,} filas, {len(columnas)} variables."
]

for archivo in archivos_test:
    prueba = pd.read_csv(archivo)

    if "class" not in prueba.columns:
        raise ValueError(f"No se encontró la etiqueta 'class' en {archivo.name}.")

    columnas_prueba = [
        c for c in prueba.columns
        if c != "class" and not c.lower().startswith("unnamed")
    ]

    if columnas_prueba != columnas:
        raise ValueError(f"Las variables de {archivo.name} no coinciden con entrenamiento.")

    y = prueba["class"].astype(int)
    if not set(y.unique()).issubset({-1, 1}):
        raise ValueError(f"Etiquetas inesperadas en {archivo.name}: {sorted(y.unique())}")

    X = prueba[columnas].apply(pd.to_numeric, errors="raise")
    X_escalado = escalador.transform(X)

    resultado = pd.DataFrame(X_escalado, columns=columnas)
    resultado["class"] = y.to_numpy()
    resultado.to_csv(SALIDA / archivo.name, index=False)

    conteo = y.value_counts().to_dict()
    resumen.append(
        f"{archivo.name}: {len(prueba):,} filas; etiquetas {conteo}."
    )

joblib.dump(escalador, SALIDA / "minmax_scaler.joblib")

print("\n".join(resumen))
print(f"Archivos preparados en: {SALIDA}")
print("Se conservaron las filas repetidas.")