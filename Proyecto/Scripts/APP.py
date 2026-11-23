# Sets de pruebas
localizacion = 'local'
dir = 'dataset/dataset.csv'

# LIBRERÍAS DEL SISTEMA
import os                  # Para manejar rutas de archivos y variables de entorno
import sys                 # Para interactuar con el sistema operativo y argumentos de ejecución
import time                # Para hacer pausas (ej. esperar a que cargue el servidor)
import urllib.request      # Para hacer peticiones web (lo usamos para revisar si Streamlit ya encendió)

# LIBRERÍAS DE MATEMÁTICAS, DATOS Y GRÁFICOS
import matplotlib.pyplot as plt  # Para dibujar gráficas (la gráfica de pastel de confianza)
import numpy as np               # Para cálculos matemáticos y arreglos numéricos eficientes
import pandas as pd              # Para leer y manipular los datos estructurados (archivos CSV, tablas)
import scipy.stats as st         # Para cálculos estadísticos (obtener el valor Z para el margen de error)

# LIBRERÍAS DE MACHINE LEARNING (SCIKIT-LEARN)
from sklearn.compose import ColumnTransformer             # Para aplicar transformaciones distintas a ciertas columnas
from sklearn.ensemble import HistGradientBoostingRegressor # Modelo GBT (Gradient Boosting) útil para datos con vacíos
from sklearn.metrics import r2_score                      # Para medir qué tan precisos son nuestros modelos
from sklearn.model_selection import train_test_split      # Para dividir los datos en entrenamiento y prueba
from sklearn.neural_network import MLPRegressor           # Modelo de Redes Neuronales Artificiales (RN)
from sklearn.pipeline import make_pipeline                # Para encadenar la limpieza de datos y el modelo en un solo paso
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler # Herramientas para traducir texto a números y escalar datos
from sklearn.svm import SVR                               # Modelo de Máquinas de Vectores de Soporte (SVR)

# LIBRERÍAS DE INTERFAZ GRÁFICA
import streamlit as face  # Framework para crear la interfaz web de la aplicación rápidamente (se apodó 'face')
import webview            # Crea una ventana de escritorio nativa que muestra una página web por dentro


# FUNCIONES DE LECTURA, LIMPIEZA Y PREPARACIÓN DE DATOS

def funLeerDS(ruta):
  # Lee el archivo CSV y lo convierte en un DataFrame de Pandas (una tabla)
  if localizacion == 'colab':
    from google.colab import drive
    drive.mount('/content/drive/')
    data = pd.read_csv(ruta)
  elif localizacion in ('kaggle', 'local'):
    data = pd.read_csv(ruta)
  else:
    raise ValueError('Localización no válida')
  return data


def funPrepVariables(ds):
  # Selecciona qué columnas vamos a usar para predecir (x) y qué queremos predecir (y)
  global atributos, objetivo, x, y
  atributos = [
      'local', 'dia_semana', 'hora', 'mes',
      'temperature_2m (°C)', 'precipitation_probability (%)', 'rain (mm)',
  ]
  objetivo = 'compras'

  x = ds[atributos].copy()
  y = ds[objetivo].copy()
  return x, y


def funLimpiar_datos(dataset):
  # Prepara los datos: quita espacios, filtra horarios y rellena datos faltantes (nulos)
  data_limpia = dataset.copy()
  data_limpia.columns = data_limpia.columns.str.strip()

  # Si existe una columna de tiempo, filtramos para que solo tome horario de 7 am a 7 pm (19 hrs)
  if 'time' in data_limpia.columns:
    data_limpia['time'] = pd.to_datetime(data_limpia['time'])
    data_limpia = data_limpia[
        (data_limpia['time'].dt.hour >= 7) & (data_limpia['time'].dt.hour <= 19)
    ].copy()
    data_limpia['mes'] = data_limpia['time'].dt.month

  # Limpiamos las columnas numéricas
  columnas_num = data_limpia.select_dtypes(include=[np.number]).columns
  for col in columnas_num:
    data_limpia[col] = data_limpia[col].astype(float)
    media = data_limpia[col].mean()
    desv = data_limpia[col].std()
    
    # Rellenamos celdas vacías con el promedio
    data_limpia[col] = data_limpia[col].fillna(media)

    # Si un dato es absurdamente alto o bajo (fuera de 3 desviaciones), lo promediamos para evitar errores
    if desv > 0:
      condicion_exagerado = (data_limpia[col] < (media - 3 * desv)) | (
          data_limpia[col] > (media + 3 * desv)
      )
      data_limpia.loc[condicion_exagerado, col] = media

  # Limpiamos columnas de texto (categóricas), rellenando vacíos con el texto más común (la moda)
  columnas_cat = data_limpia.select_dtypes(exclude=[np.number]).columns
  for col in columnas_cat:
    if col != 'time' and data_limpia[col].isnull().any():
      data_limpia[col] = data_limpia[col].fillna(data_limpia[col].mode()[0])

  return data_limpia


def funCrear_preprocesador(X):
  # Le dice al sistema cómo tratar el texto (OneHotEncoder) y cómo escalar los números (StandardScaler)
  columnas_categoricas = ['local', 'dia_semana']
  columnas_numericas = [
      col for col in X.columns if col not in columnas_categoricas
  ]

  return ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), columnas_numericas),
          (
              'cat',
              OneHotEncoder(drop='first', handle_unknown='ignore'),
              columnas_categoricas,
          ),
      ]
  )


def funEntrenar_svr(X, y):
  # Entrena el modelo de Vectores de Soporte
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  modelo_svr = make_pipeline(
      funCrear_preprocesador(X),
      SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'),
  )
  modelo_svr.fit(X_train, y_train)
  return modelo_svr


def funEntrenar_mlp(X, y):
  # Entrena el modelo de Redes Neuronales
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  modelo_mlp = make_pipeline(
      funCrear_preprocesador(X),
      MLPRegressor(
          hidden_layer_sizes=(100, 50),
          activation='relu',
          solver='adam',
          max_iter=500,
          random_state=42,
          early_stopping=True,
      ),
  )
  modelo_mlp.fit(X_train, y_train)
  return modelo_mlp


def funObtener_coeficientemlp(modelo_mlp, datos_limpios):
  # Hace la predicción en el modelo de Red neuronal
  return modelo_mlp.predict(datos_limpios)


def funEntrenarGBT(x, y):
  # Entrena el modelo GBT, el cual maneja valores de texto con OrdinalEncoder (necesario para GBT)
  columnas_categoricas = ['local', 'dia_semana']
  columnas_numericas = [
      col for col in x.columns if col not in columnas_categoricas
  ]

  preprocesador_gbt = ColumnTransformer(
      transformers=[
          ('num', 'passthrough', columnas_numericas),
          (
              'cat',
              OrdinalEncoder(
                  handle_unknown='use_encoded_value',
                  unknown_value=-1,
                  encoded_missing_value=np.nan,
              ),
              columnas_categoricas,
          ),
      ]
  )
  mGBT = make_pipeline(
      preprocesador_gbt, HistGradientBoostingRegressor(random_state=42)
  )
  mGBT.fit(x, y)
  return mGBT


# FUNCIONES ESTADÍSTICAS Y CÁLCULO DE MÁRGENES DE ERROR

def funObtener_n(dataset):
  # Devuelve la cantidad de registros en la base de datos
  return len(dataset)


def funObtener_z(nivel_confianza, n):
  # Calcula el valor estadístico Z necesario para sacar el margen de error
  alfa_medios = (1 - nivel_confianza) / 2
  if n < 30:
    return st.t.ppf(1 - alfa_medios, df=n - 1)
  return st.norm.ppf(1 - alfa_medios)


def funObtSE(y_real, y_pred):
  # Calcula el error estándar del modelo (qué tanto se equivoca en promedio)
  residuales = np.array(y_real) - np.array(y_pred)
  return np.sqrt(np.sum(residuales**2) / (len(y_real) - 2))


def funIC(prediccion, z, se):
  # Calcula el límite inferior y superior (rango mínimo y máximo de compras)
  margen = z * se
  return max(0, prediccion - margen), prediccion + margen


def funBestModelDecision(n, flagArgVacios):
  # Lógica para elegir qué modelo usar basándose en la cantidad de datos y si faltan argumentos
  if flagArgVacios == 1:
    return 'GBT'
  elif n < 500:
    return 'SVR'
  elif 500 <= n <= 10000:
    return 'GBT'
  return 'RN'


def funDeterminacion(y_real, y_pred):
  # Calcula el R2 (coeficiente de determinación), que es un % de qué tan bien explica el modelo los datos
  return r2_score(y_real, y_pred)


def readlocales(data):
  # Obtiene una lista alfabética de todos los locales disponibles en los datos
  return sorted(data['local'].unique())


def funModelUpdate(ruta_csv):
  # Lee el CSV, limpia los datos, y entrena a los 3 modelos de cero. Devuelve un diccionario con los modelos
  data_cruda = funLeerDS(ruta_csv)
  data_limpia = funLimpiar_datos(data_cruda)
  X, y = funPrepVariables(data_limpia)

  mGBT = funEntrenarGBT(X, y)
  mSVR = funEntrenar_svr(X, y)
  mRN = funEntrenar_mlp(X, y)

  return {'GBT': mGBT, 'SVR': mSVR, 'RN': mRN}, X, y


# INTERFAZ GRÁFICA CON STREAMLIT (face)

def showArgBar(locales):
  # face.columns() divide la pantalla verticalmente en 3 columnas del mismo tamaño
  col1, col2, col3 = face.columns(3)

  # with colX: coloca todo lo que esté indentado dentro de esa columna específica
  with col1:
    # face.selectbox() crea una lista desplegable (combobox)
    local_in = face.selectbox('Local / Tienda', locales)
    dia_in = face.selectbox(
        'Día de la semana',
        ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes'],
    )
    mes_in = face.selectbox('Mes', list(range(1, 13)), index=10)

  with col2:
    # face.number_input() crea una caja donde solo se pueden escribir números (puede tener mín, máx y default)
    hora_in = face.number_input('Hora (7 a 19)', 7, 19, 12)
    temp_in = face.number_input('Temperatura (°C)', value=22.0)
    confianza_in = face.selectbox(
        'Nivel de Confianza', [0.90, 0.95, 0.99], index=1
    )

  with col3:
    # face.checkbox() crea una casilla para palomear. Devuelve True si se palomeó, False si no.
    dejar_vacios = face.checkbox('Simular argumentos vacíos (Forzar GBT)', value=False)
    
    # El atributo 'disabled=dejar_vacios' desactiva la caja si el checkbox está marcado
    prob_lluvia_in = face.number_input(
        'Probabilidad de lluvia (%)',
        0.0, 100.0, 10.0,
        disabled=dejar_vacios,
    )
    lluvia_in = face.number_input(
        'Lluvia (mm)', 0.0, 100.0, 0.0, disabled=dejar_vacios
    )

  # Devolvemos un diccionario con todos los valores que el usuario introdujo
  return {
      'local': local_in,
      'dia_semana': dia_in,
      'hora': hora_in,
      'mes': mes_in,
      'temperature_2m (°C)': temp_in,
      'precipitation_probability (%)': prob_lluvia_in,
      'rain (mm)': lluvia_in,
      'confianza': confianza_in,
      'vacio': dejar_vacios,
  }


def verificarVacios(datos):
  # Si la casilla "vacio" está activada, devuelve 1 (verdadero)
  return 1 if datos['vacio'] else 0


def funPredecirFila(modelo, nombre, argumentos):
  # Convierte lo que introdujo el usuario en una fila (DataFrame) para pasársela al modelo y generar la predicción
  df_input = pd.DataFrame([{
      'local': argumentos['local'],
      'dia_semana': argumentos['dia_semana'],
      'hora': argumentos['hora'],
      'mes': argumentos['mes'],
      'temperature_2m (°C)': argumentos['temperature_2m (°C)'],
      'precipitation_probability (%)': argumentos['precipitation_probability (%)'],
      'rain (mm)': argumentos['rain (mm)'],
  }])
  if nombre == 'RN':
    return funObtener_coeficientemlp(modelo, df_input)[0]
  return modelo.predict(df_input)[0]


def funShowIC(lim_inf, lim_sup, modelo_usado):
  # face.info() dibuja un cuadro azul con información en la interfaz web
  face.info(
      f'[{modelo_usado}] Compras mínimas: {lim_inf:.1f}  |  Compras máximas: {lim_sup:.1f}'
  )


def funShowConfianza(nivel_confianza):
  # Creamos una gráfica de pastel usando Matplotlib (fondo oscuro)
  fig, ax = plt.subplots(figsize=(3.2, 3.2))
  fig.patch.set_facecolor('#121212')
  ax.set_facecolor('#121212')

  porcentaje = nivel_confianza * 100
  restante = 100 - porcentaje

  etiquetas = [f'Confianza ({porcentaje:.0f}%)', 'Margen error']
  valores = [porcentaje, restante]
  colores = ['#1e88e5', '#424242']

  ax.pie(
      valores,
      labels=etiquetas,
      autopct='%1.0f%%',
      startangle=90,
      colors=colores,
      textprops={'color': 'white', 'fontsize': 9},
  )
  ax.set_title('Nivel de confianza', color='white', fontsize=11)
  
  # face.pyplot() le indica a Streamlit que muestre en pantalla una gráfica creada con Matplotlib
  face.pyplot(fig)


def funShowReporte(nc, min_compras, max_compras, det):
  # Dependiendo del % de precisión del modelo (det), imprimimos mensajes de distintos colores
  nc_porcentaje = int(nc * 100)

  if det >= 66:
    # face.success() dibuja un cuadro VERDE (éxito)
    face.success(
        f'Dadas las condiciones hay una certeza estadística del {nc_porcentaje}% '
        f'de que el local tenga {min_compras:.1f} compras mínimas y '
        f'{max_compras:.1f} compras máximas...'
    )
  elif det >= 45:
    # face.warning() dibuja un cuadro AMARILLO (precaución)
    face.warning(
        f'Dadas las condiciones hay una certeza estadística del {nc_porcentaje}% '
        f'de que el local tenga {min_compras:.1f} compras mínimas y {max_compras:.1f} máximas, '
        'PERO HAY QUE TENER CUIDADO ya que nuestros modelos explican el '
        f'{det:.1f}% de los resultados...'
    )
  else:
    # face.error() dibuja un cuadro ROJO (error / desconfiable)
    face.error(
        f'INFLUYEN MUCHO OTROS FACTORES... resultados parcialmente desconfiables. '
        f'Para {nc_porcentaje}% de certeza, calculamos {min_compras:.1f} min y {max_compras:.1f} max.'
    )


def funShowEnviar(modelos_dict, x, y, argumentos):
  # face.button() crea un botón clickeable. Si el usuario lo presiona, el código que sigue se ejecuta
  if face.button('Enviar'):
    flag_vacios = verificarVacios(argumentos)
    nombre_mejor = funBestModelDecision(funObtener_n(x), flag_vacios)
    modelo_activo = modelos_dict[nombre_mejor]

    prediccion = funPredecirFila(modelo_activo, nombre_mejor, argumentos)
    y_pred = modelo_activo.predict(x)
    se = funObtSE(y, y_pred)
    z_critico = funObtener_z(argumentos['confianza'], funObtener_n(x))
    lim_inf, lim_sup = funIC(prediccion, z_critico, se)
    det = funDeterminacion(y, y_pred) * 100

    # face.divider() dibuja una línea horizontal gris para separar secciones
    face.divider()
    
    funShowIC(lim_inf, lim_sup, nombre_mejor)

    # Volvemos a dividir la pantalla pero ahora en 2 columnas: la izquierda más grande (2) que la derecha (1)
    col_rep, col_graf = face.columns([2, 1])
    with col_rep:
      funShowReporte(argumentos['confianza'], lim_inf, lim_sup, det)
    with col_graf:
      funShowConfianza(argumentos['confianza'])


# main

if __name__ == '__main__':
  import subprocess
  from streamlit.web import cli as stcli

  # Determinar la ruta correcta dependiendo si es código fuente o ya está compilado (.exe)
  if getattr(sys, 'frozen', False):
    script_path = os.path.join(sys._MEIPASS, os.path.basename(__file__))
  else:
    script_path = os.path.abspath(__file__)

  # Usamos una variable de entorno para saber en qué fase de arranque estamos
  if 'FASE_STREAMLIT' not in os.environ:
    
    # === FASE 1: PROCESO PRINCIPAL (Lanza Servidor en fondo y abre Ventana) ===
    env_copia = os.environ.copy()
    env_copia['FASE_STREAMLIT'] = '1'

    # Le pedimos al ejecutable que se abra a sí mismo pasándole la orden secreta 'arrancar_servidor'
    if getattr(sys, 'frozen', False):
      comando = [sys.executable, 'arrancar_servidor']
    else:
      comando = [sys.executable, script_path, 'arrancar_servidor']

    # Subprocess no tiene los bugs de multiprocessing en PyInstaller
    proceso_st = subprocess.Popen(
        comando, 
        env=env_copia,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    )

    # Esperamos a que levante el servidor web local (hasta 15 seg)
    puerto = 8501
    url_local = f'http://localhost:{puerto}'
    for _ in range(30):
      try:
        urllib.request.urlopen(url_local)
        break
      except Exception:
        time.sleep(0.5)

    # Abrimos la ventana nativa de la aplicación
    webview.create_window('Simulador Cafetería UABC', url_local, width=1280, height=800)
    webview.start()

    # Cuando el usuario cierra la ventana, matamos el servidor limpiamente
    proceso_st.terminate()
    sys.exit()

  else:
    # === FASE 2: ARRANCAR MOTOR (Subproceso invisible) ===
    # Si detecta la orden, arranca el motor interno de Streamlit
    if len(sys.argv) > 1 and sys.argv[1] == 'arrancar_servidor':
      sys.argv = [
          "streamlit", "run", script_path,
          "--server.port=8501", "--server.headless=true",
          "--global.developmentMode=false"
      ]
      sys.exit(stcli.main())

    # === FASE 3: CONSTRUCCIÓN DE LA PÁGINA (Renderizado Web) ===
    # Esta parte solo se ejecuta cuando Streamlit ya está corriendo y carga la interfaz
    
    face.set_page_config(page_title='Simulador Cafetería UABC', layout='wide')

    face.markdown(
        """
        <style>
          [data-testid="stStatusWidget"] { display: none !important; }
          .st-emotion-cache-1kyxreq { display: none !important; }
          .stApp, div, input, select, button { transition: none !important; }
          [data-testid="stAppViewContainer"] { opacity: 1 !important; filter: none !important; }
          div[data-testid="stBlock"] { opacity: 1 !important; }
          .stApp, [data-testid="stHeader"] { background-color: #121212; color: #ffffff; }
          label, p, h1, h2, h3 { color: #ffffff !important; }
          .stButton > button {
              background-color: #1e88e5 !important; color: #ffffff !important;
              border-radius: 6px; border: none; font-weight: bold; padding: 0.5rem 1.2rem;
              opacity: 1 !important;
          }
          .stButton > button:hover { background-color: #1565c0 !important; }
          div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
              background-color: #ffffff !important; color: #121212 !important; border-radius: 5px;
          }
          div[data-baseweb="input"] input, div[data-baseweb="select"] span { color: #121212 !important; }
          button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {
              background-color: #e0e0e0 !important; color: #121212 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    face.title('Simulador de Demanda - Cafetería UABC')

    if 'modelos_dict' not in face.session_state:
      modelos_dict, x, y = funModelUpdate(dir)
      face.session_state['modelos_dict'] = modelos_dict
      face.session_state['x'] = x
      face.session_state['y'] = y

    if face.button('Actualizar modelos'):
      modelos_dict, x, y = funModelUpdate(dir)
      face.session_state['modelos_dict'] = modelos_dict
      face.session_state['x'] = x
      face.session_state['y'] = y

    face.divider()

    locales = readlocales(face.session_state['x'])
    argumentos = showArgBar(locales)
    
    funShowEnviar(
        face.session_state['modelos_dict'],
        face.session_state['x'],
        face.session_state['y'],
        argumentos,
    )