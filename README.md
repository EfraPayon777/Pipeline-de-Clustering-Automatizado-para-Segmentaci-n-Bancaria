Pipeline Automatizado de Clustering para Datos Bancarios

Este proyecto implementa un flujo de trabajo (pipeline) de Machine Learning no supervisado diseñado para segmentar clientes bancarios. El sistema preprocesa datos mixtos, entrena múltiples algoritmos de clustering simultáneamente, evalúa su rendimiento mediante métricas internas y selecciona automáticamente el mejor modelo para realizar inferencias sobre nuevos datos.
Características Principales

    Preprocesamiento Híbrido: Manejo automático de variables categóricas (OneHotEncoder) y numéricas (StandardScaler) mediante ColumnTransformer.

    Evaluación Multi-Modelo: Comparación simultánea de 4 algoritmos:

        K-Means

        Gaussian Mixture Models (GMM)

        Birch

        DBSCAN (con muestreo optimizado para eficiencia).

    Selección Inteligente: Algoritmo de decisión que elige el "modelo ganador" basándose en un puntaje combinado de métricas (Silhouette, Calinski-Harabasz, Davies-Bouldin).

    Visualización PCA: Reducción de dimensionalidad a 2D para graficar la distribución de los clusters.

    Simulación de Producción: Capacidad para proyectar y clasificar "nuevos datos" (datos no vistos) sobre la estructura de clusters existente.

Tecnologías Utilizadas

    Python 3.x

    Pandas & NumPy: Manipulación de datos y álgebra lineal.

    Scikit-learn: Algoritmos de ML, preprocesamiento y métricas.

    Matplotlib: Generación de gráficos estáticos.

Instalación y Requisitos

Asegúrate de tener instaladas las dependencias necesarias:
Bash

pip install pandas numpy scikit-learn matplotlib

Uso

    Coloca tu archivo de datos bank.csv en la raíz del proyecto.

        Nota: El script espera un archivo CSV separado por punto y coma (;).

    Ejecuta el script principal:

Bash

python main.py

    Los resultados se generarán automáticamente en la carpeta outputs/.

Estructura del Pipeline

El código sigue una ejecución secuencial en 5 pasos:

    Carga y Comprensión: Lectura del dataset y análisis exploratorio básico (EDA).

    Preparación: Transformación de datos (Encoding y Scaling) para homogeneizar las características.

    Modelado: Entrenamiento de los algoritmos y cálculo de métricas de calidad de cluster (Silhouette Score, etc.).

    Selección: Ranking de modelos y exportación de métricas a CSV.

    Visualización e Inferencia:

        Generación de un panel comparativo 2x2.

        Selección de datos de prueba (simulando nuevos clientes).

        Proyección PCA y generación del gráfico final con el modelo ganador.

Resultados (Outputs)

Al finalizar la ejecución, la carpeta /outputs contendrá:

    metrics_YYYYMMDD-HHMMSS.csv: Tabla comparativa con el rendimiento de todos los modelos.

    overview_models_...png: Comparativa visual de los 4 algoritmos.

    best_model_with_new_...png: Visualización final del mejor modelo, mostrando los clusters originales y la clasificación de los nuevos datos (en color marrón).
