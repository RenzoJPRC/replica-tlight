from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODELOS = {
    "Isolation Forest": "if",
    "OCNN ReLU": "ocnn",
    "OCNN código original": "ocnn_codigo_original",
    "OCSVM": "ocsvm",
}
CONJUNTOS = [f"test_set_{i}" for i in range(1, 6)]
REPETICIONES = 2000
SEMILLA = 42


def f1_ponderado(y_real, y_pred):
    return f1_score(
        y_real,
        y_pred,
        labels=[-1, 1],
        average="weighted",
        zero_division=0,
    )


rng = np.random.default_rng(SEMILLA)
resumen = []
comparaciones = []
remuestras = []

for conjunto in CONJUNTOS:
    predicciones = {}
    y_real_referencia = None

    for nombre_modelo, prefijo in MODELOS.items():
        ruta = RESULTS / f"{prefijo}_{conjunto}_predicciones.csv"
        df = pd.read_csv(ruta)

        y_real = df["class_real"].to_numpy(dtype=int)
        y_pred = df["class_predicha"].to_numpy(dtype=int)

        if y_real_referencia is None:
            y_real_referencia = y_real
        elif not np.array_equal(y_real_referencia, y_real):
            raise ValueError(
                f"Las etiquetas reales no coinciden entre modelos en {conjunto}."
            )

        predicciones[nombre_modelo] = y_pred

    y = y_real_referencia
    n = len(y)
    puntajes = {
        modelo: np.empty(REPETICIONES)
        for modelo in MODELOS
    }

    # Se remuestrean las mismas filas para todos los modelos.
    for b in range(REPETICIONES):
        indices = rng.integers(0, n, size=n)
        y_boot = y[indices]

        for modelo, y_pred in predicciones.items():
            puntajes[modelo][b] = f1_ponderado(
                y_boot,
                y_pred[indices],
            )

    # F1 observado e intervalo bootstrap del 95%.
    f1_observado = {
        modelo: f1_ponderado(y, y_pred)
        for modelo, y_pred in predicciones.items()
    }

    for modelo in MODELOS:
        inferior, superior = np.quantile(
            puntajes[modelo], [0.025, 0.975]
        )
        resumen.append({
            "conjunto": conjunto,
            "modelo": modelo,
            "f1_observado_ponderado": f1_observado[modelo],
            "ic95_inferior": inferior,
            "ic95_superior": superior,
            "filas": n,
            "remuestras": REPETICIONES,
        })

        for b, valor in enumerate(puntajes[modelo], start=1):
            remuestras.append({
                "conjunto": conjunto,
                "remuestra": b,
                "modelo": modelo,
                "f1_ponderado": valor,
            })

    # Diferencias pareadas: F1 del primer modelo menos el segundo.
    pares = [
        ("Isolation Forest", "OCNN ReLU"),
        ("Isolation Forest", "OCNN código original"),
        ("Isolation Forest", "OCSVM"),
        ("OCNN ReLU", "OCNN código original"),
        ("OCNN ReLU", "OCSVM"),
        ("OCNN código original", "OCSVM"),
    ]

    for modelo_a, modelo_b in pares:
        diferencias = puntajes[modelo_a] - puntajes[modelo_b]
        inferior, superior = np.quantile(
            diferencias, [0.025, 0.975]
        )
        delta_observado = f1_observado[modelo_a] - f1_observado[modelo_b]

        comparaciones.append({
            "conjunto": conjunto,
            "comparacion": f"{modelo_a} - {modelo_b}",
            "diferencia_f1_observada": delta_observado,
            "ic95_inferior": inferior,
            "ic95_superior": superior,
            "diferencia_significativa_ic95": (
                inferior > 0 or superior < 0
            ),
        })

pd.DataFrame(resumen).to_csv(
    RESULTS / "bootstrap_f1_resumen.csv", index=False
)
pd.DataFrame(comparaciones).to_csv(
    RESULTS / "bootstrap_f1_comparaciones.csv", index=False
)
pd.DataFrame(remuestras).to_csv(
    RESULTS / "bootstrap_f1_remuestras.csv", index=False
)

print("\nF1 ponderado e intervalos bootstrap del 95%:")
print(pd.DataFrame(resumen).round(4).to_string(index=False))

print("\nDiferencias pareadas entre modelos:")
print(pd.DataFrame(comparaciones).round(4).to_string(index=False))

print("\nArchivos guardados en:", RESULTS)
