from pathlib import Path
import pandas as pd

PROYECTO = Path(__file__).resolve().parents[1]
archivo = PROYECTO / "data" / "test_data" / "test_set_5.csv"
datos = pd.read_csv(archivo)

variables = [
    c for c in datos.columns
    if c != "class" and not c.lower().startswith("unnamed")
]

grupos = datos.groupby(variables, dropna=False, sort=False)
cantidad_conflictos = 0

for patron, grupo in grupos:
    if grupo["class"].nunique() > 1:
        cantidad_conflictos += 1
        print(f"Patrón conflictivo #{cantidad_conflictos}")
        print(f"Registros con este patrón: {len(grupo)}")
        print("Frecuencia por etiqueta:")
        print(grupo["class"].value_counts().sort_index().to_string())
        print("Valores de las señales:")
        print(pd.Series(patron, index=variables).to_string())

if cantidad_conflictos == 0:
    print("No se encontraron patrones con etiquetas distintas.")