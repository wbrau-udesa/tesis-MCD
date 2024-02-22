from libraries import *

def langdetect(text): 
    '''
    DETECCIÓN DE IDIOMA
    
    Input:
    - text (string): un texto
    
    Output:
    - tupla donde el primer elemento es el idioma más probable 
      y el segundo la probabilidad asignada a ese idioma
    - si no logra detectar, devuelve "err"
       
    '''
    
    
    try: 
        langs = detect_langs(text) 
        for item in langs: 
            return item.lang, item.prob # The first one returned is usually the one that has the highest probability
    except: return "err", 0.0 


def subt_to_line(s):
    
    '''
    SUBTÍTULO A LÍNEA
    
    Input: s (string) es el texto con todo el subtítulo de la película
           - Restricción: por el momento para archivos .srt
    
    Output: pandas DF con dos columnas
            - cleaned_line (string): el texto de la línea
                > quitando líneas referidas a los creadores del subtítulo o la película
                > quitando las etiquetas HTML o que indican qué personaje habla
            - current_time (string): el momento en que se dice esa línea 
   '''

    # obtener la líneas
    lines = s.split('\n ')

    # obtener tupla con cada línea limpia y el momento qen que se dice 
    processed_subtitles = []
    current_time = None

    for line in lines:

        speaker_removed = False
        line = unidecode.unidecode(line.lower())


        # Extraer info sobre el tiempo ------------------------------------------------------------
        # (se asume el formato "00:01:23,456 --> 00:01:25,789")
        time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', line)
        if time_match:
            current_time = time_match.group()
            continue ## and skip it, just use that line as a time mark

        # Quitar algunas líneas  ---------------------------------------------------------------------
        if re.match(r'^[0-9]*',line).group() != '': ## comienzan con números 
            continue

        if re.match(r'(^written by|^directed by|^art direction)',line): ## listan hacedores de la película
            continue

        if any(keyword in line for keyword in ["http", "specialweirdo", "explosiveskull",
                                               "subtitle", "allsubs", "torrent", "yify",
                                               "mkvmking", "torporr", "bozxphd", "troll",
                                               "www", "yts", "dvdrip", "download", "cinemascom",
                                               "yahoocom", "decla-film", "*+1981", "p@", "@hotmail.com" 
                                              "improved by:"]): ## refieren a creadores del subtítulos
            continue


        # Quitar partes de cada línea  ---------------------------------------
        cleaned_line = re.sub(r'^[A-Za-z][A-Za-z\s]*:\s*', '', line) ## quita información de quien habla (asumiendo formato "Personaje: Hola, mundo!")
        cleaned_line = re.sub(r'<.*?>', '', cleaned_line) ## Remover etiquetasa HTML como itálicas o negritas
        cleaned_line = re.sub(r'[^\x00-\x7F]+', '', cleaned_line)
        


        # Guardar y append a la línea anterior
        if current_time and cleaned_line.strip():
            processed_subtitles.append((current_time, cleaned_line))
            
    return processed_subtitles




def get_word_vector(word, model):
        
    '''
    VECTOR NUMÉRICO POR PALABRA
    
    Input:
    - word (string): una palabra o lema
    - model (object): un modelo precargado como word2vec o Glove
    
    Output
    - vector numérico para cada word
      > si no está dentro del vocabulario del modelo, devuelve 0
    '''
    try:
        return model.get_vector(word)
    except KeyError:
        return np.zeros(300)  # Devuelve 0 para palabras OOV (out-of-vocabulary)
    


