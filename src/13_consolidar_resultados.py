from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

comparacion = pd.read_csv(
    RESULTS / "comparacion_f1_articulo.csv"
)
bootstrap = pd.read_csv(
    RESULTS / "bootstrap_f1_resumen.csv"
)

# Unificamos el nombre de OCNN para poder combinar las tablas.
comparacion["modelo"] = comparacion["modelo"].replace({
    "OCNN (ReLU)": "OCNN"
})

columnas_bootstrap = [
    "conjunto",
    "modelo",
    "ic95_inferior",
    "ic95_superior",
]
bootstrap = bootstrap[columnas_bootstrap]

tabla_final = comparacion.merge(
    bootstrap,
    on=["conjunto", "modelo"],
    how="left",
    validate="one_to_one",
)

if tabla_final["ic95_inferior"].isna().any():
    faltantes = tabla_final[
        tabla_final["ic95_inferior"].isna()
    ][["modelo", "conjunto"]]
    raise ValueError(
        "Faltan intervalos bootstrap para estas filas:\n"
        + faltantes.to_string(index=False)
    )

tabla_final = tabla_final.rename(columns={
    "f1_articulo": "f1_publicado",
    "f1_replicado": "f1_replicado_ponderado",
    "diferencia": "diferencia_replicado_menos_articulo",
})

tabla_final = tabla_final[[
    "modelo",
    "conjunto",
    "f1_publicado",
    "f1_replicado_ponderado",
    "ic95_inferior",
    "ic95_superior",
    "diferencia_replicado_menos_articulo",
]]

tabla_final.to_csv(
    RESULTS / "comparacion_final.csv",
    index=False,
    encoding="utf-8-sig",
)

# Resumen de los cinco conjuntos por modelo.
resumen = (
    tabla_final
    .groupby("modelo", as_index=False)
    .agg(
        f1_publicado_promedio=("f1_publicado", "mean"),
        f1_replicado_promedio=("f1_replicado_ponderado", "mean"),
        diferencia_promedio=("diferencia_replicado_menos_articulo", "mean"),
        diferencia_absoluta_promedio=(
            "diferencia_replicado_menos_articulo",
            lambda x: x.abs().mean(),
        ),
    )
)

resumen.to_csv(
    RESULTS / "resumen_final_por_modelo.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nComparación final por conjunto:")
print(tabla_final.round(4).to_string(index=False))

print("\nPromedios por modelo:")
print(resumen.round(4).to_string(index=False))

print("\nArchivos generados:")
print(RESULTS / "comparacion_final.csv")
print(RESULTS / "resumen_final_por_modelo.csv")