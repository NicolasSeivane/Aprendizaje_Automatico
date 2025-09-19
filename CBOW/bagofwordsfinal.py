import numpy as np

diccionario_palabras = {}
diccionario_onehot = {}

with open("corpus.txt", "r", encoding="utf-8") as f:
    words = f.read().splitlines()

for token in words:
    if token not in diccionario_palabras:

        index = len(diccionario_palabras)
        diccionario_palabras[token] = index

cardinal_V = len(diccionario_palabras)

for token, idx in list(diccionario_palabras.items()):

    one_hot_vector = np.zeros(cardinal_V)
    one_hot_vector[idx] = 1
    diccionario_onehot[token] = one_hot_vector


def softmax(u):
    u_max = np.max(u)
    e_u = np.exp(u - u_max)
    return e_u / e_u.sum()



def CBOW(diccionario_palabras, corpus, neuronas_oculta, W=None, W_prima=None, contexto=5, epocas=1000, n=0.01):
    cardinal_V = len(diccionario_palabras)

    if not W_prima and not W:
        W = np.random.uniform(0,1,(cardinal_V, neuronas_oculta))
        W_prima = np.random.uniform(0,1,(neuronas_oculta, cardinal_V))

    palabras = len(corpus)

    for i in range(epocas):

        for indice in range(contexto,(palabras-contexto)):
            
            palabra = corpus[indice]
            indice_central = diccionario_palabras[palabra]

            palabras_contexto = corpus[indice-contexto:indice] + corpus[indice+1:indice+contexto+1]
            indices_contexto = [diccionario_palabras[palabra] for palabra in palabras_contexto]

            #h = calcular_exitacion_bag(indices_contexto,W)
            h = np.mean(W[indices_contexto], axis=0).reshape(-1,1)

            u = W_prima.T@h

            y = softmax(u)

            e = y.copy()

            e[indice_central] = y[indice_central] - 1

            W_prima = W_prima -n*(h@e.T)

            EH = W_prima@e

            W = W -n*(EH.T/len(indices_contexto))

            if indice % 1000 == 0:
                print(f"termino palabra: {indice}")
  
        print(f"termino epoca: {i}")
    return W, W_prima


def CBOW_indices(diccionario_palabras, corpus, neuronas_oculta,nombre_pc, W=None, W_prima=None, contexto=5, epocas=1000, n=0.01):
    cardinal_V = len(diccionario_palabras)

    if not W_prima and not W:
        W = cp.random.normal(0,1,(cardinal_V, neuronas_oculta))
        W_prima = cp.random.normal(0,1,(neuronas_oculta, cardinal_V))
    else:
        W = cp.asarray(W)
        W_prima = cp.asarray(W_prima)


    indices = [i for i in range(contexto,(len(corpus)-contexto))]
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]
    indices_tuplas = [(diccionario_palabras[corpus[i]], [diccionario_palabras[corpus[i+j]] for j in indices_contexto]) for i in indices]

    for epoca in range(epocas):

        for i, (indice,contexto)in enumerate(indices_tuplas):
            
            h = cp.mean(W[contexto], axis=0).reshape(-1,1)

            u = W_prima.T@h

            y = softmax(u)

            e = y

            e[indice] -= 1

            W_prima -= n*(h@e.T)

            EH = W_prima@e

            W[contexto] -=n * EH.T / len(contexto)

            if i % 1000 == 0:
                print(f"termino palabra: {i}, epoca:{epoca}")
  
        print(f"termino epoca: {epoca}")
        if i % 50 == 0:
            nombre_archivo = f'pesos_cbow_{nombre_pc}_epoca{epoca}.npz'
            W1_np = cp.asnumpy(W)
            W2_np = cp.asnumpy(W_prima)
            np.savez(nombre_archivo, W1=W1_np, W2=W2_np, eta=n, N=neuronas_oculta, C=contexto)
            print(f"Pesos e hiperparámetros guardados exitosamente en '{nombre_archivo}'")
    return W, W_prima

W1,W2 = CBOW_indices(diccionario_palabras, words, 200,"pc2",contexto=5, epocas=1000, n=0.01)

import numpy as np
import cupy as cp
import random

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

def cbow_negativos_indices(diccionario_palabras, corpus, nombre_pc,
                           neuronas_oculta, n, contexto, epocas, negativos,
                           W=None, W_prima=None):

    cardinal_V = len(diccionario_palabras)
    if W is None and W_prima is None:
        W = cp.random.uniform(0, 1, (cardinal_V, neuronas_oculta))
        W_prima = cp.random.uniform(0, 1, (neuronas_oculta, cardinal_V))
    else:
        W = cp.asarray(W)
        W_prima = cp.asarray(W_prima)

    indices = [i for i in range(contexto, (len(corpus) - contexto))]
    indices_contexto = [i for i in range(-contexto, 0)] + [i for i in range(1, contexto + 1)]
    vocab_indices = list(range(len(diccionario_palabras)))

    # Cada tupla = (target, contexto, negativos)
    indices_tuplas = [
        (
            diccionario_palabras[corpus[i]],  # target central
            [diccionario_palabras[corpus[i + j]] for j in indices_contexto],  # contexto
            random.sample(
                list(set(vocab_indices) -
                     set([diccionario_palabras[corpus[i + j]] for j in indices_contexto])),
                k=negativos
            )
        )
        for i in indices
    ]

    for epoca in range(epocas):
        for i, (indice_central, contexto_idx, negativos_idx) in enumerate(indices_tuplas):

            # ---- Paso 1: hidden = promedio embeddings del contexto
            h = cp.mean(W[contexto_idx], axis=0).reshape(-1, 1)

            # ---- Paso 2: subconjunto de salida = negativos + target
            subconjunto = list(set([indice_central] + negativos_idx))
            u_sub = W_prima[:, subconjunto].T @ h  # (len(subconjunto), 1)

            # ---- Paso 3: probas sigmoides
            y = sigmoid(u_sub)

            # ---- Paso 4: error: target = 1, negativos = 0
            EL_sub = y
            pos_idx = subconjunto.index(indice_central)
            EL_sub[pos_idx] -= 1  # resta 1 solo al target positivo

            # ---- Paso 5: actualizar W_prima
            W_prima[:, subconjunto] -= n * (h @ EL_sub.T)

            # ---- Paso 6: backprop hacia embeddings de entrada
            EH = W_prima[:, subconjunto] @ EL_sub  # (dim, 1)
            
            W[contexto_idx] -= n * EH.T / len(contexto_idx)

            if i % 1000 == 0:
                print(f"termino palabra: {i}, epoca:{epoca}")

        print(f"termino epoca: {epoca}")

        # Guardado periódico
        if epoca % 50 == 0:
            nombre_archivo = f'pesos_cbow_neg_{nombre_pc}_epoca{epoca}.npz'
            W1_np = cp.asnumpy(W)
            W2_np = cp.asnumpy(W_prima)
            np.savez(nombre_archivo, W1=W1_np, W2=W2_np,
                     eta=n, N=neuronas_oculta, C=contexto, num_neg=negativos)
            print(f"Pesos guardados en '{nombre_archivo}'")

    return W, W_prima


  

