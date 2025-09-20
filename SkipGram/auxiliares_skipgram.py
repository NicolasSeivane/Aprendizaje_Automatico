import numpy as np
import cupy as cp
import random

palabras_a_indice = {}
palabras_a_onehot = {}

with open("corpus.txt", "r", encoding="utf-8") as f:
    words = f.read().splitlines()

for token in words:
    if token not in palabras_a_indice:

        index = len(palabras_a_indice)
        palabras_a_indice[token] = index

cardinal_V = len(palabras_a_indice)

for token, idx in list(palabras_a_indice.items()):

    one_hot_vector = np.zeros(cardinal_V)
    one_hot_vector[idx] = 1
    palabras_a_onehot[token] = one_hot_vector


def softmax(u):
    u_max = np.max(u)
    e_u = np.exp(u - u_max)
    return e_u / e_u.sum()

def sigmoid(x):
    return (cp.tanh(x / 2) + 1) / 2

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
