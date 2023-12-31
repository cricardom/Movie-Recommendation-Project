<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# Movie Recommendation Project 
by Ricardo Moreno
### PROYECTO INDIVIDUAL **`Machine Learning Operations (MLOps)`** (DATA SCIENCE Part-Time-01 SOY HENRY)


¡Bienvenidos al primer proyecto individual de la etapa de labs! En esta ocasión, deberán hacer un trabajo situándose en el rol de un *MLOps Engineer*!

Desarrollar una API para recomendacion de Peliculas , para esto nos suministran dos archivos .csv con los datos necesarios par mi trabajo.

## TRANSFORMACION

Comenzamos importando librerias necesarias para realizar la ingesta de datos, desencriptar y transformar las  bases de datos, importamos La Libreria Pandas, Numpy, Matplotlib, Seaborn principalmente.

Convertir en 0 o eliminar los datos nulos encontrados segun el caso, dar formato a las fechas.

Se deben eliminar las columnas que no van a ser utilizadas asi como normalizar algunas otras columnas par que no interfieran con otras.

Verificamos que el Dataset este limpio y nos permita realizar las consultas solicitadas.

Para desarrollar la API, usamos el framework *FastAPI* para crear las funciones solicitadas

El Analisis explroratorio de datos se realiza con los datos limpios para comprobar que no hay anomalias y ver patrones de analisis.

Una vez que toda la data es consumible por la API, está lista para consumir por los departamentos de Analytics y Machine Learning, y nuestro EDA nos permite entender bien los datos a los que tenemos acceso, es hora de entrenar nuestro modelo de machine learning para armar un sistema de recomendación de películas.

## EDA

 El EDA debería incluir gráficas interesantes para extraer datos, como por ejemplo una nube de palabras con las palabras más frecuentes en los títulos de las películas. Éste consiste en recomendar películas a los usuarios basándose en películas similares, por lo que se debe encontrar la similitud de puntuación entre esa película y el resto de películas, se ordenarán según el score de similaridad y devolverá una lista de Python con 5 valores, cada uno siendo el string del nombre de las películas con mayor puntaje, en orden descendente.

 Para nuestro fin definimos in histograma con la cantidad de valores no]ulos por columas,un boxplot par buscar datos atipicos en la columna budget, comprobamos correlacion entre columna de datos, histograma para popularidad por datos finalmente un scatter de 'Revenue vs. Budget'.

 ## API

 Mediante ***FastAPI*** ponemos disponibles los datos, gracias a los *decorados* mostramos las consultas:

 1 Peliculas estrenadas por Mes.
 2 Peliculas estrenadas por Dia.
 3 Score y Año de estreno por Pelicula.
 4 Votos por titulo.
 5 Exito de actor.
 6 Exito de Director.
 7 Sistema deRecomendacion de peliculas.

## Deploy

El deslpiege de nuestra API la hacemos mediante *Render.com* lo encontramos en:  https://cricardom-movie-recommendation-project.onrender.com/docs

## Github

El repositorio se encuentra en: https://github.com/cricardom/Movie-Recommendation-Project.git

## Video

El video que describe la API recomendadora de peliculas lo encontramos en youtube en este enlace:s https://youtu.be/vL9djYf0E9o


##### QUIERO AGRADECER A DIOS Y BENDECIR A TODOS LOS COMPAÑEROS, AMIGOS Y FAMILIARES QUE DE UNA U OTRA FORMA ME AYUDARON A DESARROLLAR ESTE PROYECTO! 
