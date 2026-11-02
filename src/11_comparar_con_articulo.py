from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROYECTO = Path(__file__).resolve().parents[1]
RESULTADOS = PROYECTO / "results"

comparacion = pd.read_csv(RESULTADOS / "comparacion_modelos.csv")

f1_articulo = {
    "OCSVM": [0.89, 0.84, 0.84, 0.71, 0.68],
    "OCNN (ReLU)": [0.90, 0.84, 0.86, 0.79, 0.68],
    "Isolation Forest": [0.91, 0.97, 0.87, 0.85, 0.77],
}

filas = []

for modelo, valores in f1_articulo.items():
    for i, f1_publicado in enumerate(valores, start=1):
        conjunto = f"test_set_{i}"
        fila = comparacion[
            (comparacion["modelo"] == modelo)
            & (comparacion["conjunto"] == conjunto)
        ]

        f1_replicado = float(fila["f1_ponderado"].iloc[0])

        filas.append({
            "modelo": modelo,
            "conjunto": conjunto,
            "f1_articulo": f1_publicado,
            "f1_replicado": f1_replicado,
            "diferencia": f1_replicado - f1_publicado,
        })

tabla = pd.DataFrame(filas)
tabla.to_csv(RESULTADOS / "comparacion_f1_articulo.csv", index=False)

print(tabla.round(4).to_string(index=False))

fig, ejes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
conjuntos = [f"Test {i}" for i in range(1, 6)]
x = np.arange(5)
ancho = 0.36

for eje, modelo in zip(ejes, f1_articulo):
    datos = tabla[tabla["modelo"] == modelo]
    eje.bar(
        x - ancho / 2,
        datos["f1_articulo"],
        ancho,
        label="Artículo",
    )
    eje.bar(
        x + ancho / 2,
        datos["f1_replicado"],
        ancho,
        label="Réplica",
    )
    eje.set_title(modelo)
    eje.set_xticks(x)
    eje.set_xticklabels(conjuntos, rotation=30)
    eje.set_ylim(0, 1)
    eje.grid(axis="y", alpha=0.25)

ejes[0].set_ylabel("F1 ponderado")
ejes[-1].legend()
fig.suptitle("F1 del artículo y de la réplica")
fig.tight_layout()

ruta_figura = RESULTADOS / "comparacion_f1_articulo.png"
fig.savefig(ruta_figura, dpi=160)
plt.close(fig)

print(f"\nTabla: {RESULTADOS / 'comparacion_f1_articulo.csv'}")
print(f"Gráfico: {ruta_figura}")