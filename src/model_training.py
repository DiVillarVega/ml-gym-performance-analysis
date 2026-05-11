from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier # Importamos KNN
# Importamos el preprocesador creado en el paso anterior
from src.data_preprocessing import get_preprocessor

def get_regression_pipeline(model_type='random_forest', random_state=42, knn_metric='euclidean') -> Pipeline:
    """
    Retorna un pipeline de Scikit-Learn para problemas de regresión (ej: predecir Fat_Percentage).
    Para KNN, usa knn_metric='euclidean' (Pitágoras) o 'manhattan'.
    """
    preprocessor = get_preprocessor()
    
    if model_type == 'random_forest':
        # Uso eficiente de recursos y reproducibilidad con random_state
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    elif model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'knn':
        # Agregamos KNN con la métrica seleccionada
        model = KNeighborsRegressor(n_neighbors=5, metric=knn_metric)
    else:
        raise ValueError("Modelo no soportado. Usa 'random_forest', 'linear' o 'knn'.")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    return pipeline

def get_classification_pipeline(model_type='random_forest', random_state=42, knn_metric='euclidean') -> Pipeline:
    """
    Retorna un pipeline para problemas de clasificación (ej: predecir Experience_Level).
    Para KNN, usa knn_metric='euclidean' (Pitágoras) o 'manhattan'.
    """
    preprocessor = get_preprocessor()
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == 'knn':
        # Agregamos KNN con la métrica seleccionada
        model = KNeighborsClassifier(n_neighbors=5, metric=knn_metric)
    else:
        raise ValueError("Modelo no soportado. Usa 'random_forest', 'logistic' o 'knn'.")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline