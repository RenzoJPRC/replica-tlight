# Réplica del experimento TLIGHT

Proyecto de réplica de los experimentos de detección de anomalías en el sistema TLIGHT mediante Isolation Forest, OCSVM y One-Class Neural Network (OCNN).

## Requisitos

- Python 3.11
- Entorno virtual `.venv`
- Dependencias indicadas en `requirements.txt`

Para activar el entorno e instalar las dependencias en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Estructura del proyecto

- `data/`: datos originales y preparados.
- `results/`: modelos, métricas, predicciones, comparaciones y gráficos.
- `src/`: scripts de auditoría, preparación y análisis.
- `repos/TLIGHT-SYSTEM/`: repositorio de código y datos TLIGHT consultado para la réplica.
- `requirements.txt`: versiones instaladas de las dependencias.

## Orden de ejecución

Ejecutar desde la raíz del proyecto y con `.venv` activado:

```powershell
python .\src\01_auditar_datos.py
python .\src\02_revision_patrones.py
python .\src\03_inspeccionar_conflicto.py
python .\src\04_preparar_datos.py
python .\src\05_entrenar_if.py
python .\src\06_entrenar_ocsvm.py
python .\src\07_diagnostico_ocsvm.py
python .\src\08_entrenar_ocnn.py
python .\src\08_entrenar_ocnn_codigo_original.py
python .\src\09_diagnosticar_ocnn.py
python .\src\10_comparar_modelos.py
python .\src\11_comparar_con_articulo.py
python .\src\12_bootstrap_f1.py
python .\src\13_consolidar_resultados.py
```

Los scripts `05_entrenar_if.py`, `06_entrenar_ocsvm.py`, `08_entrenar_ocnn.py` y `08_entrenar_ocnn_codigo_original.py` entrenan los modelos (incluyendo las dos variantes de OCNN). Los scripts posteriores analizan sus predicciones y resultados. Para ejecutar `12_bootstrap_f1.py`, primero deben estar guardadas en `results/` las predicciones de los cuatro modelos. Después se ejecuta `13_consolidar_resultados.py`, que consolida los resultados del bootstrap con los valores publicados en el artículo.

## Convenciones de los datos

- La etiqueta `1` representa comportamiento normal.
- La etiqueta `-1` representa una anomalía.
- Se eliminaron las columnas de índice que no eran variables del sistema.
- Se conservaron los registros repetidos porque representan estados observados en los datos.
- El conjunto de entrenamiento no incluye etiquetas; los conjuntos de prueba sí.

## Resultados principales

Los resultados consolidados se guardan en:

- `results/comparacion_final.csv`: F1 publicado, F1 replicado e intervalo bootstrap por modelo y conjunto.
- `results/resumen_final_por_modelo.csv`: promedios por modelo.
- `results/bootstrap_f1_comparaciones.csv`: diferencias pareadas entre modelos.
- `results/comparacion_f1_articulo.png`: gráfico comparativo frente a los valores publicados.

## Alcance estadístico

El bootstrap pareado compara cuatro modelos (`Isolation Forest`, `OCNN ReLU`, `OCNN código original` y `OCSVM`) mediante 2,000 remuestras pareadas de las filas de cada conjunto, manteniendo las mismas filas remuestreadas para todos los modelos. Sus intervalos del 95% describen la variabilidad asociada a esos conjuntos y predicciones, y sus comparaciones pareadas no tienen ajuste por comparaciones múltiples. No representan nuevos entrenamientos ni datos de otros sistemas.

El artículo menciona ANOVA y Tukey sobre 20 muestras, pero no especifica completamente cómo se generaron. Por ello, el análisis bootstrap de esta réplica es complementario y no debe presentarse como reproducción exacta de esas pruebas.

## Limitaciones de la réplica

- OCNN se adaptó a TensorFlow 2 para poder ejecutarse en el entorno actual.
- Se fijó una semilla para Isolation Forest porque el artículo no especifica la utilizada.
- Los valores F1 publicados están redondeados, y algunas diferencias entre modelos y conjuntos son amplias.

## Reproducir desde cero en Windows

```powershell
git clone https://github.com/RenzoJPRC/replica-tlight.git
Set-Location .\replica-tlight
```

Ejecutar estos pasos desde la carpeta raíz del proyecto en PowerShell:

### 1. Crear el entorno e instalar dependencias

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

### 2. Descargar y preparar los datos

Los archivos de datos se obtienen del repositorio público TLIGHT-SYSTEM. Estos comandos descargan el repositorio y extraen los CSV a `data/`:

```powershell
New-Item -ItemType Directory -Force .\repos, .\data | Out-Null
git clone https://github.com/emmanuelaboah/TLIGHT-SYSTEM.git .\repos\TLIGHT-SYSTEM
Expand-Archive -Path .\repos\TLIGHT-SYSTEM\Dataset\train_data.zip -DestinationPath .\data -Force
Expand-Archive -Path .\repos\TLIGHT-SYSTEM\Dataset\test_data.zip -DestinationPath .\data -Force
```

### 3. Ejecutar el flujo de la réplica

Con el entorno virtual activado, ejecutar en orden los scripts indicados en la sección **Orden de ejecución**.
## Analisis adicional: variantes OCNN e histogramas

### Comparacion de variantes OCNN

Se conservaron dos implementaciones para hacer explicita una diferencia del codigo fuente disponible:

- `src/08_entrenar_ocnn.py` usa la activacion ReLU.
- `src/08_entrenar_ocnn_codigo_original.py` conserva el comportamiento literal del codigo publicado: aunque la funcion auxiliar se llama `relu`, devuelve la entrada sin aplicar ReLU.

La segunda variante fue adaptada para TensorFlow 2. Ambas se comparan con los valores publicados mediante `src/14_comparar_variantes_ocnn.py`. Los resultados muestran diferencias entre conjuntos; ninguna variante reproduce exactamente todos los valores del articulo.

Para repetir esta comparacion, primero ejecutar los entrenamientos de ambas variantes y luego:

```powershell
python .\src\14_comparar_variantes_ocnn.py
```

### Histogramas de puntajes de decision

El script `src/15_generar_histogramas.py` crea 20 imagenes en `results/plots`: una por cada combinacion de modelo y conjunto de prueba. Requiere las predicciones guardadas de Isolation Forest, OCSVM y ambas variantes OCNN.

```powershell
python .\src\15_generar_histogramas.py
```

La visualizacion esta adaptada del repositorio Decision-Score-Histogram. Su licencia MIT se conserva en `src/LICENSE_Decision-Score-Histogram.txt`. La guia `results/plots/README.md` explica la convencion de clases y la interpretacion de los ejes.
