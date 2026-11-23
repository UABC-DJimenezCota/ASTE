# Drive
from google.colab import drive
drive.mount('/content/drive')

# Imports necesarios
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def funCargar_y_filtrar_datos(ruta_csv):
    """
    Carga el dataset, filtra por horario operativo y extrae características temporales.
    Retorna los predictores (X) y la variable objetivo (y).
    """
    # Cargar datos
    data = pd.read_csv(ruta_csv)
    
    # Limpiar nombres de columnas
    data.columns = data.columns.str.strip()
    
    # Conversión y filtrado
    data['time'] = pd.to_datetime(data['time'])
    data_limpia = data[(data['time'].dt.hour >= 7) & (data['time'].dt.hour <= 19)].copy()
    
    # Características temporales
    data_limpia['mes'] = data_limpia['time'].dt.month
    data_limpia['dia_mes'] = data_limpia['time'].dt.day
    
    # Separar X e y
    y = data_limpia['compras']
    X = data_limpia.drop(columns=['compras', 'time'])
    
    print(f"Total de datos procesados: {len(data_limpia)}")
    return X, y

def funCrear_preprocesador(X):
    """
    Crea el pipeline de preprocesamiento (OneHotEncoding para categorías, StandardScaler para números).
    """
    columnas_categoricas = ['local', 'dia_semana']
    columnas_numericas = [col for col in X.columns if col not in columnas_categoricas]
    
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), columnas_categoricas)
        ]
    )
    return preprocesador

def funEntrenar_svr(X, y):
    """
    Entrena un modelo SVR y evalúa su rendimiento.
    """
    print("\n--- Entrenando SVR ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo_svr = make_pipeline(
        funCrear_preprocesador(X),
        SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale')
    )
    
    modelo_svr.fit(X_train, y_train)
    y_pred = modelo_svr.predict(X_test)
    
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    
    return modelo_svr

def funEntrenar_mlp(X, y):
    """
    Entrena un modelo MLPRegressor (Red Neuronal) y evalúa su rendimiento.
    """
    print("\n--- Entrenando MLPRegressor ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2 capas ocultas con 100 y 50 neuronas
    modelo_mlp = make_pipeline(
        funCrear_preprocesador(X),
        MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                     max_iter=500, random_state=42, early_stopping=True)
    )
    
    modelo_mlp.fit(X_train, y_train)
    y_pred = modelo_mlp.predict(X_test)
    
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    
    return modelo_mlp

def funObtener_coeficientemlp(modelo_mlp, datos_limpios):
    """
    Recibe el modelo entrenado de Redes Neuronales (MLPRegressor) y los datos limpios (argumentos).
    Retorna la predicción calculada por el modelo.
    """
    prediccion = modelo_mlp.predict(datos_limpios)
    return predicción

def funObtener_n(dataset):
    """
    Recibe el dataset y retorna la cantidad total de registros.
    """
    n = len(dataset)
    return n

def funObtener_z(nivel_confianza, n):
    """
    Recibe el nivel de confianza (ej. 0.95 para 95%) y N.
    Retorna el valor crítico Z.
    """
    if n < 30:
        # Si la muestra es pequeña, se recomienda usa la distribución T de Student
        grados_libertad = n - 1
        alfa_medios = (1 - nivel_confianza) / 2
        z_o_t = st.t.ppf(1 - alfa_medios, df=grados_libertad)
    else:
        # Para muestras grandes, se usa la distribución Normal Estándar
        alfa_medios = (1 - nivel_confianza) / 2
        z_o_t = st.norm.ppf(1 - alfa_medios)
        
    return z_o_t
