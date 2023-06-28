#importamos las librerias requeridas
import pandas as pd
import numpy as np
from fastapi import FastAPI 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.extmath import randomized_svd
from sklearn.feature_extraction.text import TfidfVectorizer
import zipfile
import uvicorn
import csv


app = FastAPI(title='Movie Recommendation Project    by   Ricardo Moreno',
              description='API para recomendar Peliculas, PROYECTO SOY HENRY DATA SCIENCE PT-01',
            )

# Variables Globales
df_highly_rated = None
cv = None
count_matrix = None
cosine_sim = None
indices = None

@app.get('/')
async def read_root():
    return {'Movie Recommendation Project by Ricardo Moreno'}

@app.get('/')
async def index():
    return{'API para recomendar Peliculas'}

@app.get('/about/')
async def about():
    return {'Proyecto Individual de Data Science Part-Time 01 2023 SOY HENRY'}

 # Sacamos el Data frame zipiado
def extract_data_from_zip(zip_file):
   
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('movies.zip')
    # Lectura de data frame
    movies_df = pd.read_csv('movies/movies.csv')
    return movies_df


# Función de películas por mes
@app.get('/peliculas_mes/({mes})')
def peliculas_mes(mes:str):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes'''
     
    mes = mes.lower()
    meses = {
    'enero': 1,
    'febrero': 2,
    'marzo': 3,
    'abril': 4,
    'mayo': 5,
    'junio': 6,
    'julio': 7,
    'agosto': 8,
    'septiembre': 9,
    'octubre': 10,
    'noviembre': 11,
    'diciembre': 12}

    mes_numero = meses[mes]

    # Convertir la columna "fecha" a un objeto de tipo fecha
    df['release_date'] = pd.to_datetime(df['release_date'])

    # Tratamos la excepciòn
    try:
        month_filtered = df[df['release_date'].dt.month == mes_numero]
    except (ValueError, KeyError, TypeError):
        return None

    # Filtramos valores duplicados del dataframe y calculamos
    month_unique = month_filtered.drop_duplicates(subset='id')
    respuesta = month_unique.shape[0]

    return {'mes':mes, 'cantidad de peliculas':respuesta}

# Función de películas por día
@app.get('/peliculas_dia/({dia})')
def peliculas_dia(dia:str):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes'''
        
    days = {'lunes': 'Monday', 'martes': 'Tuesday', 'miercoles': 'Wednesday', 'jueves': 'Thursday', 'viernes': 'Friday', 'sabado': 'Saturday', 'domingo': 'Sunday'}
    day = days[dia.lower()]
    lista_peliculas_day = df[df['release_date'].dt.day_name() == day].drop_duplicates(subset='id')
    respuesta = lista_peliculas_day.shape[0]

    return {'dia': dia, 'cantidad de peliculas': respuesta}

#Funcion Score por Pelicula
@app.get('/score_titulo/{pelicula}')
def score_titulo(pelicula):
    '''Se ingresa el titulo de la pelicula y la funcion retorna el titulo, el año de estreno y el score'''
    pelicula = pelicula.lower()

    info_pelicula = df[df['title'].str.lower() == pelicula].drop_duplicates(subset='title')

    pelicula_nombre = info_pelicula['title'].iloc[0]
    year_pelicula = str(info_pelicula['release_year'].iloc[0])
    score_pelicula = info_pelicula['vote_average'].iloc[0]

    return {'pelicula': pelicula_nombre, 'año': year_pelicula, "score": score_pelicula}

#Funcion Votos por Titulo
@app.get('/votos_titulo/{pelicula}')
def votos_titulo(pelicula):
    "' Se ingresa el título de una y la funcion retorna el título, la cantidad de votos y el valor promedio de las votaciones. La misma variable deberá de contar con al menos 2000 valoraciones, caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.'"
    
    pelicula = pelicula.lower()

    info_pelicula = df[df['title'].str.lower() == pelicula].drop_duplicates(subset='title')

    pelicula_nombre = info_pelicula['title'].iloc[0]
    vote_pelicula = info_pelicula['vote_count'].iloc[0]
    score_pelicula = info_pelicula['vote_average'].iloc[0]
    if vote_pelicula < 2000:
        return 'La Pelicula debe tener mas de 2000 valoraciones'
    return {'pelicula': pelicula_nombre, 'votos': vote_pelicula, "score": score_pelicula}

# Funcion Exito Actor
@app.get('/nombre_actor/{actor}')
def get_actor(actor):
    "'Se ingresa el nombre de un actor y la funcion devuelve el éxito del mismo medido a través la cantidad de películas que en las que ha participado y el promedio de retorno.'"

    actor = actor.lower()
   
    lista_get_actor = df[df['nombre_actor'].str.lower() == actor].drop_duplicates(subset='id')
    cantidad = (lista_get_actor).shape[0]
    retorno = (lista_get_actor['return'].iloc[0])

    promedio_retorno = int(retorno)*10000000/int(cantidad)

    return {'actor':actor, 'cantidad peliculas':cantidad, 'promedio retorno': promedio_retorno}

#Funcion Exito Director
@app.get('/nombre_director/{director}')
def get_director(director:str):
    "'Se ingresa el nombre de un director y la funcion devuelve el éxito del mismo medido a través del nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.'"

    director = director.lower()

    lista_get_director = df[df['nombre_director'].str.lower() == director].drop_duplicates(subset='id')
    retorno = lista_get_director['return'].sum() / lista_get_director['budget'].sum()
    pelicula = lista_get_director[['title', 'release_year', 'return', 'budget', 'revenue']].to_dict('records')
    
    return {'director': director, 'retorno': retorno,'pelicula,': pelicula}

#Funcion de Recomendacion de Peliculas
def calculate_reduced_similarity_matrix(df):
    # El Vectorizador TfidfVectorizer con parámetros de reduccion procesamiento
    df['genres'].fillna('', inplace=True)
    vectorizar = TfidfVectorizer(min_df=10, max_df=0.5, ngram_range=(1,2))

    # Vectorizamos, ajustamos y transformamos el texto de la columna "title" del DataFrame
    X = vectorizar.fit_transform(df['genres'])

    # Calcular la matriz de similitud de coseno con una matriz reducida de 7000
    similarity_matrix = cosine_similarity(X[:1250,:])

    # Obtenemos la descomposición en valores singulares aleatoria de la matriz de similitud de coseno con 10 componentes
    n_components = 10
    U, Sigma, VT = randomized_svd(similarity_matrix, n_components=n_components)

    # Construir la matriz reducida de similitud de coseno
    reduced_similarity_matrix = U.dot(np.diag(Sigma)).dot(VT)

    return reduced_similarity_matrix
# Inicializar y cargar el DataFrame df
df = pd.read_csv('./movies.csv')

# Calcular la matriz de similitud reducida
reduced_similarity_matrix = calculate_reduced_similarity_matrix(df)

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
        
    titulo = titulo.title()
    #Ubicamos el indice del titulo pasado como parametro en la columna 'title' del dts user_item
    indice = np.where(df['title'] == titulo)[0][0]
    #Vemos los indices de aquellas puntuaciones y caracteristicas similares hacia el titulo 
    puntuaciones_similitud = reduced_similarity_matrix[indice,:]
    # Se ordena los indicies de menor a mayor
    puntuacion_ordenada = np.argsort(puntuaciones_similitud)[::-1]
    # Que solo 5 nos indique 
    top_indices = puntuacion_ordenada[:5]
    
    return df.loc[top_indices, 'title'].tolist()