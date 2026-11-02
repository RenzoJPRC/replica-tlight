# Histogramas de puntajes de decisión

La visualización está adaptada del repositorio Decision-Score-Histogram de Emmanuel Aboah Boateng. Se conserva la licencia MIT en `src/LICENSE_Decision-Score-Histogram.txt`.

## Convención de clases

El código del histograma considera `+1` (normal) como la clase positiva y `-1` (anomalía) como la clase negativa. Por eso, en estas figuras:

- True Positive: normal clasificado como normal.
- True Negative: anomalía clasificada como anomalía.
- False Positive: anomalía clasificada como normal.
- False Negative: normal clasificado como anomalía.

Esta convención difiere de las métricas del proyecto, donde la anomalía (`-1`) se trata como clase positiva. Al interpretar las figuras, un False Positive del histograma equivale a una anomalía no detectada según esas métricas.

## Puntajes y eje vertical

El cero es el umbral de decisión: los puntajes no negativos predicen normal (`+1`) y los negativos predicen anomalía (`-1`). Los puntajes positivos y negativos se normalizan por separado para mostrar su distribución y confianza. Por ello, el eje horizontal no conserva la escala original de cada modelo.

El eje vertical representa la proporción de predicciones dentro de cada clase real, no el número absoluto de registros.
