# _Repositorio para la tesis de la Maestría en Ciencia de Datos de UdeSA_
## Inmigración en películas: un análisis utilizando aprendizaje automático a partir de los subtítulos

Este repositorio contiene buena parte del código que se utilizó para la tesis. Incluye los siguientes iPython Notebooks correspondientes a distintas partes del proceso, en el siguiente orden:

1. Datos. 
    - **scraping_yify.ipynb**: scrapea y guarda subtítulos.
    
2. Limpieza y preprocesamiento
    - **preprocesamiento_subt.ipynb**: limpieza y preprocesamiento inicial de los subtítulos.

3. Análisis
    - **explora_inicial.ipynb** Este Notebook realiza una comparación inicial entre los subtítulos de las películas de inmigración y una muestra de los subtítulos de las películas de no inmigración basada en la construcción de dos puntajes de la cantidad de palabras asociadas a la inmigración presentes en las películas
    - **clustering_1_armado.ipynb**: armado de las temáticas de inmigración.
    - **clustering_2_valor-peli.ipynb**: matriz F2C (valores de las películas en las temáticas de inmigración) y modelos de regresión.
    - **clasif_f2v.ipynb**: modelos de clasificación a partir de matriz F2V para construir índice de contenido de inmigración en películas.
    - **clasif_roberta**: RoBERTa para construir índice de contenido de inmigración en películas (se corrió desde Google Colab usando T4 GPU).

Además, incluye módulos con funciones, que se importan en los Notebooks correspondientes:
- **libraries.py**: importa las librerías a usar durante todo el análisis (generales de Python)
- **limpieza_subt.py**: funciones usadas durante la limpieza de los subtítulos (armadas para este trabajo)
- **clustering.py**: clases y funciones para el análisis de clústers (armadas para este trabajo)



