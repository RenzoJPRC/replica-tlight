from pathlib import Path
import pandas as pd

PROYECTO = Path(__file__).resolve().parents[1]
RESULTADOS = PROYECTO / "results"

ARCHIVO_RELU = RESULTADOS / "ocnn_metricas.csv"
ARCHIVO_ORIGINAL = RESULTADOS / "ocnn_codigo_original_metricas.csv"
ARCHIVO_SALIDA = RESULTADOS / "comparacion_variantes_ocnn.csv"

F1_ARTICULO = {
    "test_set_1": 0.90,
    "test_set_2": 0.84,
    "test_set_3": 0.86,
    "test_set_4": 0.79,
    "test_set_5": 0.68,
}


def cargar_metricas(ruta: Path, nombre: str) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    datos = pd.read_csv(ruta)

    columnas_necesarias = {"conjunto", "f1_ponderado"}
    faltantes = columnas_necesarias - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en {ruta.name}: {faltantes}")

    datos = datos[datos["conjunto"].str.startswith("test_set_")].copy()
    datos = datos[["conjunto", "f1_ponderado"]]
    return datos.rename(columns={"f1_ponderado": nombre})


def main():
    relu = cargar_metricas(ARCHIVO_RELU, "f1_ocnn_relu")
    original = cargar_metricas(
        ARCHIVO_ORIGINAL,
        "f1_ocnn_codigo_original",
    )

    comparacion = relu.merge(original, on="conjunto", validate="one_to_one")
    comparacion["f1_articulo"] = comparacion["conjunto"].map(F1_ARTICULO)

    comparacion["diferencia_relu"] = (
        comparacion["f1_ocnn_relu"] - comparacion["f1_articulo"]
    )
    comparacion["diferencia_codigo_original"] = (
        comparacion["f1_ocnn_codigo_original"] - comparacion["f1_articulo"]
    )

    comparacion = comparacion[
        [
            "conjunto",
            "f1_articulo",
            "f1_ocnn_relu",
            "diferencia_relu",
            "f1_ocnn_codigo_original",
            "diferencia_codigo_original",
        ]
    ]

    comparacion.to_csv(ARCHIVO_SALIDA, index=False)

    print(comparacion.round(4).to_string(index=False))
    print(
        "\nError absoluto medio frente al artículo:"
        f"\n  OCNN con ReLU: "
        f"{comparacion['diferencia_relu'].abs().mean():.4f}"
        f"\n  Código original imitado: "
        f"{comparacion['diferencia_codigo_original'].abs().mean():.4f}"
    )
    print(f"\nArchivo guardado en: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()
