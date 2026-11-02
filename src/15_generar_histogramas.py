from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hist_score_tlight import AnomalyScoreHist

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
PLOTS = RESULTS / "plots"

MODELS = {
    "if": "Isolation Forest",
    "ocsvm": "OCSVM",
    "ocnn": "OCNN_ReLU",
    "ocnn_codigo_original": "OCNN_codigo_original",
}


def main():
    input_files = []

    for prefix in MODELS:
        for number in range(1, 6):
            path = RESULTS / f"{prefix}_test_set_{number}_predicciones.csv"
            if not path.exists():
                raise FileNotFoundError(f"No se encontro el archivo: {path}")
            input_files.append((prefix, number, path))

    PLOTS.mkdir(parents=True, exist_ok=True)

    original_directory = Path.cwd()
    os.chdir(RESULTS)

    try:
        for prefix, number, path in input_files:
            data = pd.read_csv(path)

            required = {"class_real", "decision_score"}
            missing = required - set(data.columns)
            if missing:
                raise ValueError(
                    f"Faltan columnas en {path.name}: {missing}"
                )

            scores = data["decision_score"].to_numpy(dtype=float)
            labels = data["class_real"].to_numpy(dtype=int)

            figure_name = f"hist_{MODELS[prefix]}_test_set_{number}"
            print(f"Generando {figure_name}...")

            AnomalyScoreHist(scores, labels).plot_hist(figure_name)
            plt.close("all")
    finally:
        os.chdir(original_directory)

    print(f"\nSe generaron {len(input_files)} histogramas en: {PLOTS}")


if __name__ == "__main__":
    main()
