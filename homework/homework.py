# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# Importación de librerías necesarias para el análisis de datos y machine learning
from sklearn.pipeline import Pipeline  # Para crear pipelines de transformación y modelado
from sklearn.compose import ColumnTransformer  # Para aplicar diferentes transformaciones a diferentes columnas
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler  # Para codificación one-hot y escalado de datos
from sklearn.feature_selection import SelectKBest, f_classif  # Para seleccionar las mejores características
from sklearn.model_selection import GridSearchCV  # Para optimización de hiperparámetros con validación cruzada
from sklearn.linear_model import LogisticRegression  # Modelo de regresión logística
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Métricas de evaluación
import gzip  # Para comprimir archivos
import pickle  # Para serializar y guardar modelos
import zipfile  # Para trabajar con archivos comprimidos
import pandas as pd  # Para manipulación de datos
import os  # Para operaciones del sistema de archivos
import json  # Para trabajar con archivos JSON


def limpiar_datos(datos):
    """
    Función para limpiar y preprocesar los datos del dataset.
    
    Esta función realiza las siguientes operaciones de limpieza:
    1. Crea una copia del dataset para evitar modificar el original
    2. Elimina la columna 'ID' que no es relevante para el modelo
    3. Renombra la columna objetivo para facilitar su uso
    4. Elimina registros con valores faltantes
    5. Filtra registros con valores no válidos en EDUCATION y MARRIAGE
    6. Agrupa niveles superiores de educación en la categoría 'others'
    
    Args:
        datos (pd.DataFrame): Dataset original a limpiar
        
    Returns:
        pd.DataFrame: Dataset limpio y procesado
    """
    # Crear una copia del dataset para evitar modificar el original
    datos = datos.copy()
    
    # Eliminar la columna ID ya que no aporta información predictiva
    datos = datos.drop('ID', axis=1)
    
    # Renombrar la columna objetivo para facilitar su manejo
    datos = datos.rename(columns={'default payment next month': 'default'})
    
    # Eliminar registros con valores faltantes (NaN)
    datos = datos.dropna()
    
    # Filtrar registros con valores no válidos (0 = N/A) en EDUCATION y MARRIAGE
    datos = datos[(datos['EDUCATION'] != 0 ) & (datos['MARRIAGE'] != 0)]
    
    # Agrupar niveles superiores de educación (>4) en la categoría 'others' (4)
    datos.loc[datos['EDUCATION'] > 4, 'EDUCATION'] = 4

    return datos

def modelo():
    """
    Función para crear el pipeline de machine learning.
    
    Este pipeline incluye las siguientes etapas:
    1. Preprocesamiento de datos (codificación y escalado)
    2. Selección de características más relevantes
    3. Modelo de regresión logística
    
    Returns:
        Pipeline: Pipeline completo de sklearn listo para entrenar
    """
    # Definir las variables categóricas que necesitan codificación one-hot
    categoricas = ['SEX', 'EDUCATION', 'MARRIAGE']  
    
    # Definir las variables numéricas que necesitan escalado
    numericas = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", 
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5","PAY_AMT6"
    ]
    
    # Crear el preprocesador que aplica diferentes transformaciones según el tipo de variable
    preprocesador = ColumnTransformer(
        transformers=[
            # Aplicar OneHotEncoder a variables categóricas (convierte categorías en variables binarias)
            ('cat', OneHotEncoder(handle_unknown='ignore'), categoricas),
            # Aplicar MinMaxScaler a variables numéricas (escala valores entre 0 y 1)
            ('num', MinMaxScaler(), numericas)
        ],
        remainder='passthrough'  # Mantener otras columnas sin cambios
    )

    # Crear el selector de características que elegirá las K mejores variables
    # Usa f_classif como función de puntuación para problemas de clasificación
    seleccion_k_mejor = SelectKBest(score_func=f_classif)

    # Crear el pipeline completo con todas las etapas
    pipeline = Pipeline(steps=[
        # Etapa 1: Preprocesamiento (codificación one-hot + escalado)
        ('preprocessor', preprocesador),
        # Etapa 2: Selección de características más relevantes
        ("selectkbest", seleccion_k_mejor),
        # Etapa 3: Modelo de regresión logística
        ('classifier', LogisticRegression(
            max_iter=1000,      # Máximo número de iteraciones para convergencia
            solver="saga",      # Algoritmo de optimización que soporta regularización L1 y L2
            random_state=42     # Semilla para reproducibilidad
        ))
    ])

    return pipeline

def hiperparametros(modelo, n_splits, x_entrenamiento, y_entrenamiento, puntuacion):
    """
    Función para optimizar los hiperparámetros del modelo usando validación cruzada.
    
    Esta función utiliza GridSearchCV para encontrar la mejor combinación de hiperparámetros
    probando diferentes valores y evaluándolos mediante validación cruzada.
    
    Args:
        modelo (Pipeline): Pipeline de machine learning a optimizar
        n_splits (int): Número de particiones para la validación cruzada (10)
        x_entrenamiento (pd.DataFrame): Características de entrenamiento
        y_entrenamiento (pd.Series): Variable objetivo de entrenamiento
        puntuacion (str): Métrica de evaluación ('balanced_accuracy')
        
    Returns:
        GridSearchCV: Modelo optimizado con los mejores hiperparámetros
    """
    # Crear el objeto GridSearchCV para búsqueda exhaustiva de hiperparámetros
    estimador = GridSearchCV(
        estimator=modelo,  # El pipeline a optimizar
        # Definir la grilla de hiperparámetros a probar
        param_grid = {
            # Número de características a seleccionar (de 1 a 10)
            "selectkbest__k": range(1, 11),
            # Tipo de regularización: L1 (Lasso) o L2 (Ridge)
            "classifier__penalty": ["l1", "l2"],
            # Fuerza de regularización (valores más altos = más regularización)
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        },
        cv=n_splits,                    # Número de particiones para validación cruzada
        refit=True,                     # Re-entrenar con mejores parámetros en todo el dataset
        verbose=0,                      # No mostrar progreso detallado
        return_train_score=False,       # No devolver puntuaciones de entrenamiento
        scoring=puntuacion              # Métrica a optimizar
    )
    
    # Entrenar el modelo probando todas las combinaciones de hiperparámetros
    estimador.fit(x_entrenamiento, y_entrenamiento)

    return estimador

def metricas(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    """
    Función para calcular las métricas de evaluación del modelo.
    
    Calcula precision, precision balanceada, recall y f1-score para los conjuntos
    de entrenamiento y prueba. Estas métricas son fundamentales para evaluar
    el rendimiento de un modelo de clasificación.
    
    Args:
        modelo (GridSearchCV): Modelo entrenado y optimizado
        x_entrenamiento (pd.DataFrame): Características de entrenamiento
        y_entrenamiento (pd.Series): Variable objetivo de entrenamiento
        x_prueba (pd.DataFrame): Características de prueba
        y_prueba (pd.Series): Variable objetivo de prueba
        
    Returns:
        tuple: (métricas_entrenamiento, métricas_prueba) como diccionarios
    """
    # Generar predicciones para el conjunto de entrenamiento
    y_entrenamiento_pred = modelo.predict(x_entrenamiento)
    
    # Generar predicciones para el conjunto de prueba
    y_prueba_pred = modelo.predict(x_prueba)

    # Calcular métricas para el conjunto de entrenamiento
    metricas_entrenamiento = {
        'type': 'metrics',
        'dataset': 'train',
        # Precision: proporción de predicciones positivas que fueron correctas
        'precision': (precision_score(y_entrenamiento, y_entrenamiento_pred, average='binary')),
        # Precision balanceada: promedio de sensibilidad y especificidad
        'balanced_accuracy':(balanced_accuracy_score(y_entrenamiento, y_entrenamiento_pred)),
        # Recall (sensibilidad): proporción de casos positivos correctamente identificados
        'recall': (recall_score(y_entrenamiento, y_entrenamiento_pred, average='binary')),
        # F1-score: media armónica entre precision y recall
        'f1_score': (f1_score(y_entrenamiento, y_entrenamiento_pred, average='binary'))
    }

    # Calcular métricas para el conjunto de prueba
    metricas_prueba = {
        'type': 'metrics',
        'dataset': 'test',
        # Precision: proporción de predicciones positivas que fueron correctas
        'precision': (precision_score(y_prueba, y_prueba_pred, average='binary')),
        # Precision balanceada: promedio de sensibilidad y especificidad
        'balanced_accuracy':(balanced_accuracy_score(y_prueba, y_prueba_pred)),
        # Recall (sensibilidad): proporción de casos positivos correctamente identificados
        'recall': (recall_score(y_prueba, y_prueba_pred, average='binary')),
        # F1-score: media armónica entre precision y recall
        'f1_score': (f1_score(y_prueba, y_prueba_pred, average='binary'))
    }

    return metricas_entrenamiento, metricas_prueba

def matriz(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    """
    Función para calcular las matrices de confusión del modelo.
    
    La matriz de confusión muestra la distribución de predicciones correctas e incorrectas,
    permitiendo analizar qué tipos de errores comete el modelo.
    
    Args:
        modelo (GridSearchCV): Modelo entrenado y optimizado
        x_entrenamiento (pd.DataFrame): Características de entrenamiento
        y_entrenamiento (pd.Series): Variable objetivo de entrenamiento
        x_prueba (pd.DataFrame): Características de prueba
        y_prueba (pd.Series): Variable objetivo de prueba
        
    Returns:
        tuple: (matriz_entrenamiento, matriz_prueba) como diccionarios
    """
    # Generar predicciones para el conjunto de entrenamiento
    y_entrenamiento_pred = modelo.predict(x_entrenamiento)
    
    # Generar predicciones para el conjunto de prueba
    y_prueba_pred = modelo.predict(x_prueba)

    # Calcular matriz de confusión para el conjunto de entrenamiento
    mc_entrenamiento = confusion_matrix(y_entrenamiento, y_entrenamiento_pred)
    # Extraer los valores de la matriz: VN=Verdaderos Negativos, FP=Falsos Positivos, 
    # FN=Falsos Negativos, VP=Verdaderos Positivos
    vn_entrenamiento, fp_entrenamiento, fn_entrenamiento, vp_entrenamiento = mc_entrenamiento.ravel()

    # Calcular matriz de confusión para el conjunto de prueba
    mc_prueba = confusion_matrix(y_prueba, y_prueba_pred)
    # Extraer los valores de la matriz
    vn_prueba, fp_prueba, fn_prueba, vp_prueba = mc_prueba.ravel()

    # Formatear matriz de confusión para entrenamiento según el formato requerido
    matriz_entrenamiento = {
        'type': 'cm_matrix',
        'dataset': 'train', 
        'true_0': {  # Casos reales negativos (no default)
            'predicted_0': int(vn_entrenamiento),  # Correctamente predichos como negativos
            'predicted_1': int(fp_entrenamiento)   # Incorrectamente predichos como positivos
        },
        'true_1': {  # Casos reales positivos (default)
            'predicted_0': int(fn_entrenamiento),  # Incorrectamente predichos como negativos
            'predicted_1': int(vp_entrenamiento)   # Correctamente predichos como positivos
        }
    }

    # Formatear matriz de confusión para prueba según el formato requerido
    matriz_prueba = {
        'type': 'cm_matrix',
        'dataset': 'test', 
        'true_0': {  # Casos reales negativos (no default)
            'predicted_0': int(vn_prueba),  # Correctamente predichos como negativos
            'predicted_1': int(fp_prueba)   # Incorrectamente predichos como positivos
        },
        'true_1': {  # Casos reales positivos (default)
            'predicted_0': int(fn_prueba),  # Incorrectamente predichos como negativos
            'predicted_1': int(vp_prueba)   # Correctamente predichos como positivos
        }
    }

    return matriz_entrenamiento, matriz_prueba

def guardar_modelo(modelo):
    """
    Función para guardar el modelo entrenado en formato comprimido.
    
    Guarda el modelo optimizado en un archivo comprimido usando gzip para
    reducir el espacio de almacenamiento. El modelo se serializa usando pickle.
    
    Args:
        modelo (GridSearchCV): Modelo entrenado y optimizado a guardar
    """
    # Crear el directorio 'files/models' si no existe
    os.makedirs('files/models', exist_ok=True)

    # Abrir archivo comprimido para escritura y guardar el modelo
    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        # Serializar y guardar el modelo usando pickle
        pickle.dump(modelo, f)

def guardar_metricas(metricas):
    """
    Función para guardar las métricas y matrices de confusión en formato JSON.
    
    Guarda todas las métricas calculadas (precision, recall, f1-score, etc.) y
    las matrices de confusión en un archivo JSON con formato de líneas separadas.
    
    Args:
        metricas (list): Lista de diccionarios con métricas y matrices de confusión
    """
    # Crear el directorio 'files/output' si no existe
    os.makedirs('files/output', exist_ok=True)

    # Abrir archivo para escritura de métricas
    with open("files/output/metrics.json", "w") as f:
        # Iterar sobre cada métrica en la lista
        for metrica in metricas:
            # Convertir cada diccionario a formato JSON
            linea_json = json.dumps(metrica)
            # Escribir cada métrica en una línea separada
            f.write(linea_json + "\n")


# ===============================================================================
# EJECUCIÓN PRINCIPAL DEL PIPELINE DE MACHINE LEARNING
# ===============================================================================

# ---- PASO 1: CARGA DE DATOS ----
# Cargar el conjunto de datos de prueba desde el archivo comprimido
with zipfile.ZipFile('files/input/test_data.csv.zip', 'r') as zip:
    with zip.open('test_default_of_credit_card_clients.csv') as doc:
        datos_prueba = pd.read_csv(doc)

# Cargar el conjunto de datos de entrenamiento desde el archivo comprimido
with zipfile.ZipFile('files/input/train_data.csv.zip', 'r') as zip:
    with zip.open('train_default_of_credit_card_clients.csv') as doc:
        datos_entrenamiento = pd.read_csv(doc)

# ---- PASO 2: LIMPIEZA Y PREPROCESAMIENTO ----
# Aplicar la función de limpieza a los datos de prueba
datos_prueba = limpiar_datos(datos_prueba)

# Aplicar la función de limpieza a los datos de entrenamiento
datos_entrenamiento = limpiar_datos(datos_entrenamiento)

# ---- PASO 3: DIVISIÓN DE VARIABLES ----
# Separar características (X) y variable objetivo (y) para el conjunto de entrenamiento
x_entrenamiento, y_entrenamiento = datos_entrenamiento.drop('default', axis=1), datos_entrenamiento['default']

# Separar características (X) y variable objetivo (y) para el conjunto de prueba
x_prueba, y_prueba = datos_prueba.drop('default', axis=1), datos_prueba['default']

# ---- PASO 4: CREACIÓN DEL MODELO ----
# Crear el pipeline de machine learning con preprocesamiento y modelo
pipeline_modelo = modelo()

# ---- PASO 5: OPTIMIZACIÓN DE HIPERPARÁMETROS ----
# Optimizar hiperparámetros usando validación cruzada con 10 particiones
# Se usa 'balanced_accuracy' como métrica para manejar posibles desequilibrios en las clases
pipeline_modelo = hiperparametros(pipeline_modelo, 10, x_entrenamiento, y_entrenamiento, 'balanced_accuracy')

# ---- PASO 6: GUARDADO DEL MODELO ----
# Guardar el modelo optimizado en formato comprimido
guardar_modelo(pipeline_modelo)

# ---- PASO 7: EVALUACIÓN DEL MODELO ----
# Calcular métricas de evaluación para ambos conjuntos de datos
metricas_entrenamiento, metricas_prueba = metricas(pipeline_modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)

# Calcular matrices de confusión para ambos conjuntos de datos
matriz_entrenamiento, matriz_prueba = matriz(pipeline_modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)

# ---- PASO 8: GUARDADO DE RESULTADOS ----
# Guardar todas las métricas y matrices de confusión en un archivo JSON
guardar_metricas([metricas_entrenamiento, metricas_prueba, matriz_entrenamiento, matriz_prueba])