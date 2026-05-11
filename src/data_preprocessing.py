import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_gym_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset de gimnasio con validación de existencia y contenido.
    
    Args:
        filepath (str): Ruta al archivo CSV.
    Returns:
        pd.DataFrame: Datos cargados o None si hay error.
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("El archivo CSV está vacío.")
        return df
    except (FileNotFoundError, ValueError) as e:
        print(f"Error técnico en carga: {e}")
        return None

def get_preprocessor() -> ColumnTransformer:
    """
    Crea el motor de transformación para variables del gimnasio.
    Asegura la escalabilidad y modularidad del pipeline.
    
    Returns:
        ColumnTransformer: Pipeline de preprocesamiento configurado.
    """
    numeric_features = [
        'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
        'Resting_BPM', 'Session_Duration (hours)', 'Water_Intake (liters)', 
        'Workout_Frequency (days/week)', 'BMI'
    ]
    categorical_features = ['Gender', 'Workout_Type']

    # Preprocesamiento modular: Escalado para numéricas y OneHot para categóricas
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])