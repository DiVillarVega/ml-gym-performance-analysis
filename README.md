# Proyecto de Modelado de Datos: Análisis de Rendimiento en Gimnasio

Este proyecto contiene el ciclo completo de desarrollo de una solución de Machine Learning para predecir y agrupar el rendimiento físico de usuarios de un gimnasio. Se aplican técnicas de aprendizaje supervisado (regresión y clasificación) y no supervisado (clustering y reducción de dimensionalidad).

Desarrollado para la evaluación parcial N°2 de Programación para la Ciencia de Datos (SCY1101).

## 📁 Estructura del Proyecto

El repositorio está organizado de la siguiente manera para separar la lógica de procesamiento de la exploración visual:

*   `data/`: Contiene el dataset original `gym_members_exercise_tracking.csv`.
*   `notebooks/`: Notebooks ejecutables numerados según el flujo lógico del proyecto (del 01 al 05).
*   `src/`: Código fuente modular (funciones de preprocesamiento, entrenamiento y evaluación).
*   `models/trained_models/`: Modelos entrenados y optimizados serializados en formato `.pkl`.
*   `results/`: Almacena los gráficos (`plots/`) generados durante los análisis.

## ⚙️ Requisitos Previos (Windows 10/11)

Asegúrate de tener instalado Python (recomendado 3.10 o superior) en tu sistema. Se recomienda ejecutar el proyecto dentro de un entorno virtual para evitar conflictos con otras librerías.

Las dependencias principales del proyecto son:
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `jupyter` o `ipykernel`

## 🚀 Guía de Instalación y Ejecución

Abre tu terminal (Símbolo del sistema o PowerShell) en la carpeta raíz del proyecto (`proyecto_modelado`) y sigue estos pasos:

**1. Crear un entorno virtual:**
```cmd
python -m venv venv
```
**2. Activar el entorno virtual:**
```cmd
.\venv\Scripts\activate
```
**3. Instalar dependencias:**
```cmd
pip install -r requirements.txt
```
**4. OPCIONAL: Ejecutar proyecto:**
Levanta el servidor de Jupyter Notebook ejecutando
```cmd
jupyter notebook
```