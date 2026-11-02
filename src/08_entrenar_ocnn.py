from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

PROYECTO = Path(__file__).resolve().parents[1]
DATOS = PROYECTO / "data" / "processed"
RESULTADOS = PROYECTO / "results"
RESULTADOS.mkdir(parents=True, exist_ok=True)

# Semilla fija para poder repetir esta ejecución.
tf.keras.utils.set_random_seed(42)

X_train_df = pd.read_csv(DATOS / "train.csv")
columnas = list(X_train_df.columns)
X_train = X_train_df.to_numpy(dtype=np.float32)

NU = 0.04
NODOS_OCULTOS = 32
TASA_APRENDIZAJE = 0.0001
EPOCAS = 100

# Pesos de la capa de entrada y de salida.
W = tf.Variable(
    tf.random.normal(
        shape=(X_train.shape[1], NODOS_OCULTOS),
        mean=0.0,
        stddev=1e-5,
        dtype=tf.float32,
    )
)
V = tf.Variable(
    tf.random.normal(
        shape=(NODOS_OCULTOS, 1),
        mean=0.0,
        stddev=1e-5,
        dtype=tf.float32,
    )
)

optimizador = tf.keras.optimizers.SGD(
    learning_rate=TASA_APRENDIZAJE
)

X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)

def obtener_puntuaciones(X):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    capa_oculta = tf.nn.relu(tf.matmul(X, W))
    return tf.matmul(capa_oculta, V)

# El artículo optimiza primero los pesos y actualiza el umbral
# con el percentil correspondiente a nu.
umbral = 0.1

print("Entrenando OCNN...")
for epoca in range(1, EPOCAS + 1):
    umbral_tf = tf.constant(umbral, dtype=tf.float32)

    with tf.GradientTape() as cinta:
        puntuaciones_train = obtener_puntuaciones(X_train_tf)

        regularizacion = (
            0.5 * tf.reduce_sum(tf.square(W))
            + 0.5 * tf.reduce_sum(tf.square(V))
        )
        perdida_hinge = (
            (1.0 / NU)
            * tf.reduce_mean(tf.nn.relu(umbral_tf - puntuaciones_train))
        )
        perdida = regularizacion + perdida_hinge - umbral_tf

    gradientes = cinta.gradient(perdida, [W, V])
    optimizador.apply_gradients(zip(gradientes, [W, V]))

    puntuaciones_actuales = obtener_puntuaciones(X_train_tf).numpy().ravel()
    umbral = float(np.percentile(puntuaciones_actuales, 100 * NU))

    if epoca == 1 or epoca % 10 == 0:
        print(
            f"Época {epoca:3d}/{EPOCAS} | "
            f"pérdida={float(perdida.numpy()):.6f} | "
            f"umbral={umbral:.8f}"
        )

def calcular_metricas(y_real, y_pred):
    precision_p, recall_p, f1_p, _ = precision_recall_fscore_support(
        y_real,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metricas = {
        "accuracy": accuracy_score(y_real, y_pred),
        "precision_ponderada": precision_p,
        "recall_ponderado": recall_p,
        "f1_ponderado": f1_p,
    }

    if len(np.unique(y_real)) > 1:
        precision_a, recall_a, f1_a, _ = precision_recall_fscore_support(
            y_real,
            y_pred,
            average="binary",
            pos_label=-1,
            zero_division=0,
        )
        metricas.update({
            "precision_anomalias": precision_a,
            "recall_anomalias": recall_a,
            "f1_anomalias": f1_a,
        })

    return metricas

filas_metricas = []

# En entrenamiento todos los registros se consideran normales (+1).
scores_train = obtener_puntuaciones(X_train_tf).numpy().ravel() - umbral
pred_train = np.where(scores_train >= 0, 1, -1)
y_train = np.ones(len(X_train), dtype=int)

filas_metricas.append({
    "conjunto": "Entrenamiento",
    **calcular_metricas(y_train, pred_train),
})

for archivo in sorted(DATOS.glob("test_set_*.csv")):
    datos = pd.read_csv(archivo)
    y_real = datos["class"].astype(int).to_numpy()
    X_test = datos[columnas].to_numpy(dtype=np.float32)

    scores = obtener_puntuaciones(X_test).numpy().ravel() - umbral
    pred = np.where(scores >= 0, 1, -1)

    filas_metricas.append({
        "conjunto": archivo.stem,
        **calcular_metricas(y_real, pred),
    })

    pd.DataFrame({
        "class_real": y_real,
        "class_predicha": pred,
        "decision_score": scores,
    }).to_csv(
        RESULTADOS / f"ocnn_{archivo.stem}_predicciones.csv",
        index=False,
    )

np.savez_compressed(
    RESULTADOS / "ocnn_pesos_y_umbral.npz",
    W=W.numpy(),
    V=V.numpy(),
    umbral=np.array(umbral),
)

tabla = pd.DataFrame(filas_metricas)
tabla.to_csv(RESULTADOS / "ocnn_metricas.csv", index=False)

print("\nResultados de OCNN:")
print(tabla.round(4).to_string(index=False))
print(f"\nModelo y resultados guardados en: {RESULTADOS}")