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

# Asignamos los valores publicados de OCNN a ambas variantes de OCNN.
mapa_modelo_articulo = {
    "Isolation Forest": "Isolation Forest",
    "OCNN ReLU": "OCNN (ReLU)",
    "OCNN código original": "OCNN (ReLU)",
    "OCSVM": "OCSVM",
}

bootstrap["modelo_articulo"] = bootstrap["modelo"].map(mapa_modelo_articulo)

if bootstrap["modelo_articulo"].isna().any():
    modelos_desconocidos = bootstrap.loc[
        bootstrap["modelo_articulo"].isna(), "modelo"
    ].unique()
    raise ValueError(
        f"Modelos sin correspondencia en el artículo: {modelos_desconocidos}"
    )

publicados = comparacion[["modelo", "conjunto", "f1_articulo"]].rename(
    columns={
        "modelo": "modelo_articulo",
        "f1_articulo": "f1_publicado",
    }
)

tabla_final = bootstrap.merge(
    publicados,
    on=["modelo_articulo", "conjunto"],
    how="left",
    validate="many_to_one",
)

if tabla_final["f1_publicado"].isna().any():
    faltantes_pub = tabla_final[
        tabla_final["f1_publicado"].isna()
    ][["modelo", "conjunto"]]
    raise ValueError(
        "Faltan valores publicados para estas filas:\n"
        + faltantes_pub.to_string(index=False)
    )

if tabla_final["ic95_inferior"].isna().any():
    faltantes_ic = tabla_final[
        tabla_final["ic95_inferior"].isna()
    ][["modelo", "conjunto"]]
    raise ValueError(
        "Faltan intervalos bootstrap para estas filas:\n"
        + faltantes_ic.to_string(index=False)
    )

tabla_final = tabla_final.rename(columns={
    "f1_observado_ponderado": "f1_replicado_ponderado",
})

tabla_final["diferencia_replicado_menos_articulo"] = (
    tabla_final["f1_replicado_ponderado"] - tabla_final["f1_publicado"]
)

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
    .groupby("modelo", as_index=False, sort=False)
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

print(f"\nFilas en comparacion_final.csv: {len(tabla_final)}")
print(f"Modelos en resumen_final_por_modelo.csv: {len(resumen)}")
print(f"Valores publicados vacíos: {int(tabla_final['f1_publicado'].isna().sum())}")

print("\nComparación final por conjunto:")
print(tabla_final.round(4).to_string(index=False))

print("\nPromedios por modelo:")
print(resumen.round(4).to_string(index=False))

print("\nArchivos generados:")
print(RESULTS / "comparacion_final.csv")
print(RESULTS / "resumen_final_por_modelo.csv")
