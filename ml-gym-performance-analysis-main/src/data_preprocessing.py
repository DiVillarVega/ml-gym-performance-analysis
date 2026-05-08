import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_gym_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset de gimnasio con manejo de excepciones.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta {filepath}")
        return None

def get_preprocessor() -> ColumnTransformer:
    """
    Crea un ColumnTransformer para preprocesar variables numéricas y categóricas.
    Excluye explícitamente variables objetivo como 'Fat_Percentage' o 'Experience_Level'.
    """
    # Definimos las columnas según el dataset
    numeric_features = [
        'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
        'Resting_BPM', 'Session_Duration (hours)', 'Water_Intake (liters)', 
        'Workout_Frequency (days/week)', 'BMI'
    ]
    
    categorical_features = ['Gender', 'Workout_Type']

    # Escalamos variables numéricas
    numeric_transformer = StandardScaler()

    # Codificamos variables categóricas
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Ensamblamos el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor