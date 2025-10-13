import numpy as np
#import cupy as cp
import random

palabras_a_indice = {}
indices_a_palabras = {}
diccionario_onehot = {}
diccionario_onehot_a_palabra = {}
diccionario_conteo = {}

with open("C:\\Users\\User\\Documents\\GitHub\\Aprendizaje_Automatico\\Evaluacion_Modelos\\corpus_junto1.txt", "r", encoding="utf-8") as f:
    words = f.read().splitlines()


def generar_negativos_pool(tuplas_base, vocab_indices, probs, k_neg, pool_factor=2):
    print("Empezo tuplas nuevas negativas")
    print("="*40)
    """
    Genera negativos usando un pool rotatorio para todas las tuplas de una época.
    tuplas_base: lista de (palabra_central, contexto)
    vocab_indices: array de índices de vocabulario
    probs: array de probabilidades normalizadas
    k_neg: cantidad de negativos por tupla
    pool_factor: cuánto más grande que la cantidad total de negativos
    """
    n_tuplas = len(tuplas_base)
    pool_size = n_tuplas * k_neg * pool_factor
    pool_negativos = np.random.choice(vocab_indices, size=pool_size, p=probs, replace=True)
    
    cursor = 0
    negativos_tuplas = []

    for palabra_central, contexto in tuplas_base:
        excluidos = set([palabra_central] + contexto)
        negativos = []

        while len(negativos) < k_neg:
            candidato = pool_negativos[cursor]
            cursor += 1
            if candidato not in excluidos:
                negativos.append(candidato)
        negativos_tuplas.append((palabra_central, contexto, negativos))
    print("Termino tuplas nuevas negativas")
    return negativos_tuplas



def softmax(u):
    u_max = np.max(u)
    e_u = np.exp(u - u_max)
    return (e_u / e_u.sum()),e_u.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    print("Se terminó tuplas originales")
    return indices_tuplas


'''def generar_tuplas_nuevo_negativos(corpus, palabra_a_indice, contexto, distribucion, k_neg=10):
 
    indices = range(contexto, len(corpus) - contexto)
    offsets_contexto = [j for j in range(-contexto, 0)] + [j for j in range(1, contexto + 1)]

    tuplas = []
    contexto_a_central = {}

    vocabulario = list(distribucion.keys())

    for i in indices:
        palabra_central = palabra_a_indice[corpus[i]]
        contexto_actual = [palabra_a_indice[corpus[i + j]] for j in offsets_contexto]

        # Evitar tuplas duplicadas con el mismo contexto
        t_contexto = tuple(contexto_actual)
        if t_contexto in contexto_a_central and contexto_a_central[t_contexto] != palabra_central:
            continue
        contexto_a_central[t_contexto] = palabra_central

        excluir = [palabra_central] + contexto_actual
        negativos = obtener_negativas_distribucion(vocabulario, distribucion, excluir, k=k_neg)

        tuplas.append((palabra_central, contexto_actual, negativos))

    return tuplas'''

def generar_distribucion_negativa(corpus, alpha=0.75):
    """Genera distribución P(w) ∝ f(w)^alpha normalizada para negative sampling."""
    frec = {}
    for palabra in corpus:
        frec[palabra] = frec.get(palabra, 0) + 1
    
    total = sum(f ** alpha for f in frec.values())
    dist = {palabra: (f ** alpha) / total for palabra, f in frec.items()}
    return dist


def actualizar_negativos_tuplas(tuplas_base, distribucion, k_neg, palabras_a_indice):
    print("Se emepezó tuplas negativas")
    vocab_palabras = list(distribucion.keys())
    vocab_indices = np.array([palabras_a_indice[p] for p in vocab_palabras])
    probs = np.array(list(distribucion.values()))
    probs = probs / probs.sum()

    negativos_tuplas = []

    for palabra_central, contexto in tuplas_base:
        excluir = set([palabra_central] + contexto)


        mask = np.isin(vocab_indices, list(excluir), invert=True)
        candidatos = vocab_indices[mask]
        probs_validas = probs[mask]
        probs_validas = probs_validas / probs_validas.sum()

        negativos = np.random.choice(candidatos, size=k_neg, replace=False, p=probs_validas)

        negativos_tuplas.append((palabra_central, contexto, negativos.tolist()))
    print("Se terminó tuplas negativas")
    return negativos_tuplas


def unir_palabras_en_contexto(corpus, palabra_objetivo1, palabra_objetivo2, contexto=1):
    corpus_modificado = []
    i = 0

    while i < len(corpus):
        palabra = corpus[i]

        # Si encontramos una de las palabras objetivo
        if palabra == palabra_objetivo1 or palabra == palabra_objetivo2:
            # Miramos dentro del rango del contexto (siguiente palabra incluida)
            for j in range(1, contexto + 1):
                if i + j < len(corpus):
                    siguiente = corpus[i + j]

                    if {palabra, siguiente} == {palabra_objetivo1, palabra_objetivo2}:
                        # Respetar orden según aparición
                        token = f"{palabra} {siguiente}"
                        corpus_modificado.append(token)
                        i += j + 1  
                        break
            else:
                
                corpus_modificado.append(palabra)
                i += 1
        else:
            corpus_modificado.append(palabra)
            i += 1

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

for token in words:
    if token not in palabras_a_indice:
        index = len(palabras_a_indice)
        palabras_a_indice[token] = index
        indices_a_palabras[index] = token
        diccionario_conteo[token] = 1 
    else:
        diccionario_conteo[token] += 1 


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
