import numpy as np
import cupy as cp
import random

palabras_a_indice = {}
indices_a_palabras = {}
diccionario_onehot = {}
diccionario_onehot_a_palabra = {}
diccionario_conteo = {}

with open("C:\\Users\\PIA\\Documents\\AprendizajeAutomatico\\Aprendizaje_Automatico-main\\Aprendizaje_Automatico-main\\corpus.txt", "r", encoding="utf-8") as f:
    words = f.read().splitlines()




def softmax(u):
    u_max = cp.max(u)
    e_u = cp.exp(u - u_max)
    return (e_u / e_u.sum()),e_u.sum()

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

def generar_tuplas(corpus, palabras_a_indice, contexto):

    ## Genero una lista de todos los indices de las palabras en el corpus, posibles sin padding
    indices = [i for i in range(contexto,(len(corpus)-contexto))]

    ## Genero una lista de los contextos de las palabras de contexto
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]

    indices_tuplas = []

    ## Por cada indice en indices, genero una tupla (indice_central, [indices_contexto])
    for i in indices:

        indices_tuplas.append((palabras_a_indice[corpus[i]], [palabras_a_indice[corpus[i+j]] for j in indices_contexto]))

    return indices_tuplas      

def generar_tuplas_nuevo(corpus, palabras_a_indice, contexto):

    indices = [i for i in range(contexto,(len(corpus)-contexto))]

    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]

    indices_tuplas = []

    contexto_a_central = {} 

    for i in indices:

        palabra_central = palabras_a_indice[corpus[i]]

        contexto_actual = tuple(palabras_a_indice[corpus[i+j]] for j in indices_contexto)

        if contexto_actual in contexto_a_central:
            if contexto_a_central[contexto_actual] != palabra_central:
                continue 
        else:
            contexto_a_central[contexto_actual] = palabra_central

        indices_tuplas.append((palabra_central, list(contexto_actual)))

    return indices_tuplas



def unir_palabras_en_contexto(corpus, palabra_objetivo1, palabra_objetivo2, contexto=1):

    corpus_modificado = corpus.copy()  
    
    for i in range(contexto, len(corpus) - contexto):

        if corpus[i] == palabra_objetivo1 or corpus[i] == palabra_objetivo2:

            inicio = i - contexto
            fin = i + contexto + 1
            
            contexto_palabra = corpus[inicio:fin]
            
            if palabra_objetivo1 in contexto_palabra and palabra_objetivo2 in contexto_palabra:
                # Si ambas están, las unimos en un solo token
                token_combinado = f"{palabra_objetivo1} {palabra_objetivo2}"
                
                for j in range(inicio, fin):
                    if corpus[j] == palabra_objetivo1 or corpus[j] == palabra_objetivo2:
                        corpus_modificado[j] = token_combinado
    
    return corpus_modificado

palabras_a_unir = [
                    (',','y'),
                    ('.','no'),
                    ('. ','no'),
                    ('que', 'lo'),
                    ('de','la'),
                    ('a','la'),
                    ('en','el'),
                    ('pero',','),
                    ('lo','que'),
                    ('la','de'),
                    ('con',','),
                    ('se','que'),
                    ('como', 'un'),
                    ('por','la'),
                    ('o',','),
                    ('no','?'),
                    ('para','que'),
                    (';','el'),
                    ('la',':')
]

corpus_modificado = words.copy()

for palabra1, palabra2 in palabras_a_unir:
    
    corpus_modificado = unir_palabras_en_contexto(corpus_modificado, palabra1, palabra2, 1)

for token in corpus_modificado:
    if token not in palabras_a_indice:
        index = len(palabras_a_indice)
        palabras_a_indice[token] = index
        indices_a_palabras[index] = token
        diccionario_conteo[token] = 1 # Initialize count for new token
    else:
        diccionario_conteo[token] += 1 # Increment count for existing token


cardinal_V = len(palabras_a_indice)

for token, idx in list(palabras_a_indice.items()):

    one_hot_vector = np.zeros(cardinal_V)
    one_hot_vector[idx] = 1
    diccionario_onehot[token] = one_hot_vector
    diccionario_onehot_a_palabra[str(one_hot_vector)] = token

def generar_tuplas_con_negativos(corpus, palabras_a_indice, contexto, num_negativos):

    ## Genero una lista de todos los indices de las palabras en el corpus, posibles sin padding
    indices = [i for i in range(contexto,(len(corpus)-contexto))]

    ## Genero una lista de los contextos de las palabras de contexto
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]


    indices_tuplas = []

    ## Por cada indice en indices, genero una tupla (indice_central, [indices_contexto], [indices_negativos])
    for i in indices:

        indices_tuplas.append(
            (palabras_a_indice[corpus[i]],  # target central
            [palabras_a_indice[corpus[i + j]] for j in indices_contexto],  # contexto
            obtener_negativas(corpus,i,contexto,num_negativos)))# negativos

    return indices_tuplas

def obtener_negativas(corpus, indice, contexto, num_negativos=5):
    # Hasta dónde mirar (contexto + margen de negativas)
    maximo = contexto + num_negativos // 2

    # Ventana de contexto
    inicio = max(0, indice - maximo)
    fin = min(len(corpus), indice + maximo + 1)

    # Negativas candidatas izquierda y derecha
    izq = corpus[max(0, inicio - maximo):inicio]
    der = corpus[fin:min(len(corpus), fin + maximo)]

    # Balanceo: si falta en izquierda, saco más de derecha (y viceversa)
    if len(izq) < num_negativos // 2:
        faltan = (num_negativos // 2) - len(izq)
        der = corpus[fin:min(len(corpus), fin + maximo + faltan)]

    elif len(der) < num_negativos // 2:
        faltan = (num_negativos // 2) - len(der)
        izq = corpus[max(0, inicio - maximo - faltan):inicio]

    # Filtrar: excluir la palabra central
    candidatas = [p for p in izq + der if p != corpus[indice]]

    # Convertir a índices y devolver solo las necesarias
    return [palabras_a_indice[p] for p in candidatas[:num_negativos]]


def generar_tuplas_con_negativos_random(corpus, palabras_a_indice, contexto, num_negativos):

    ## Genero una lista de todos los indices de las palabras en el corpus, posibles sin padding
    indices = [i for i in range(contexto,(len(corpus)-contexto))]

    ## Genero una lista de los contextos de las palabras de contexto
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]
    vocabulario = list(set(range(len(palabras_a_indice))))

    indices_tuplas = []

    ## Por cada indice en indices, genero una tupla (indice_central, [indices_contexto], [indices_negativos])
    for i in indices:

        indices_tuplas.append(
            (palabras_a_indice[corpus[i]],  # target central
            [palabras_a_indice[corpus[i + j]] for j in indices_contexto],  # contexto
            random.sample(vocabulario - set([palabras_a_indice[corpus[i + j]] for j in indices_contexto]), k=num_negativos)))# negativos

    return indices_tuplas

def cargar_modelo_completo(nombre_archivo='pesos_cbow_pc2_epoca0.npz'):
    """
    Carga los pesos W1, W2 y los hiperparámetros N, C y eta
    desde un archivo .npz.
    """
    try:
        data = np.load(nombre_archivo)

        W1 = data['W1']
        W2 = data['W2']

        N = data['N'].item()
        C = data['C'].item()
        eta = data['eta'].item()

        print()

        return W1, W2, N, C, eta

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo}'.")
        return None, None, None, None, None
