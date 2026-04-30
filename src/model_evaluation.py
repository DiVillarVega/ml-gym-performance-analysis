from sklearn.model_selection import cross_validate
import pandas as pd

def evaluate_model_cv(pipeline, X, y, task='regression', cv=5):
    """
    Realiza validación cruzada robusta devolviendo múltiples métricas.
    """
    if task == 'regression':
        scoring = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']
    elif task == 'classification':
        scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    else:
        raise ValueError("Task debe ser 'regression' o 'classification'.")

    # Ejecutamos cross_validate
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)
    
    # Formateamos los resultados en un DataFrame para fácil lectura
    results_df = pd.DataFrame(scores).mean().to_frame(name='Mean_CV_Score')
    return results_df