#import numpy as np
import random
import numpy as cp

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

    one_hot_vector = cp.zeros(cardinal_V)
    one_hot_vector[idx] = 1
    diccionario_onehot[token] = one_hot_vector


def softmax(u):
    e_u = cp.exp(u)
    return e_u / cp.sum()


def skip_gram(diccionario_palabras, corpus, neuronas_oculta, n, contexto, epocas, W=None, W_prima=None):
    cardinal_V = len(diccionario_palabras)
    if W is None and W_prima is None:

        W = cp.random.normal(0,1,(cardinal_V, neuronas_oculta))
        W_prima = cp.random.normal(0,1,(neuronas_oculta, cardinal_V))

    cardinal_corpus = len(corpus)

    for i in range(epocas):
        for indice in range(contexto-1,(cardinal_corpus-contexto)+1):
            
            palabra = corpus[indice]
            indice_central = diccionario_palabras[palabra]

            palabras_contexto = corpus[indice-contexto:indice] + corpus[indice+1:indice+contexto+1]
            indices_contexto = [diccionario_palabras[palabra] for palabra in palabras_contexto]

            h = W[indice_central].reshape(-1,1)

            u = W_prima.T@h

            y = softmax(u)


            #EL = calcular_ej_skipgram(y,indices_contexto)
            EL = y.copy()

            EL[indices_contexto] -= 1

            W_prima -= n * (h @ EL.T)

            EH = W_prima @ EL

            W[indice_central] -= n * EH.T[0]
            print("LLegamos bien carajo")
  
        print(f"termino epoca: {i}")
    return W, W_prima



def skip_gram_indices(diccionario_palabras, corpus, neuronas_oculta,nombre_pc, n, contexto, epocas, W=None, W_prima=None):
    cardinal_V = len(diccionario_palabras)
    if W is None and W_prima is None:
        
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

            h = W[indice].reshape(-1,1)

            u = W_prima.T@h

            y = softmax(u)


            #EL = calcular_ej_skipgram(y,indices_contexto)
            EL = y

            EL[contexto] -= 1

            W_prima -= n * (h @ EL.T)

            EH = W_prima @ EL

            W[indice] -= n * EH.T[0]
            if i % 1000 == 0:
                print(f"termino palabra: {i}, epoca:{epoca}")
  
        print(f"termino epoca: {epoca}")

        if i % 50 == 0:
            nombre_archivo = f'pesos_cbow_{nombre_pc}_epoca{epoca}.npz'
            #W1_np = cp.asnumpy(W)
            #W2_np = cp.asnumpy(W_prima)
            cp.savez(nombre_archivo, W1=W, W2=W_prima, eta=n, N=neuronas_oculta, C=len(contexto)/2)
            print(f"Pesos e hiperparámetros guardados exitosamente en '{nombre_archivo}'")
    return W, W_prima

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

def skip_gram_negativos_indices(diccionario_palabras, corpus, nombre_pc, neuronas_oculta, n, contexto, epocas, negativos, W=None, W_prima=None):

    cardinal_V = len(diccionario_palabras)
    if W is None and W_prima is None:
        W = cp.random.uniform(0, 1, (cardinal_V, neuronas_oculta))
        W_prima = cp.random.uniform(0, 1, (neuronas_oculta, cardinal_V))
    else:
        W = cp.asarray(W)
        W_prima = cp.asarray(W_prima)

    # Precomputar vocabulario
    vocab_indices = set(range(cardinal_V))

    # Preparar los índices del corpus
    indices = [i for i in range(contexto, len(corpus) - contexto)]
    indices_contexto_offset = [i for i in range(-contexto, 0)] + [i for i in range(1, contexto + 1)]

    # Construir tuplas de entrenamiento (central, contexto, negativos)
    indices_tuplas = []
    for i in indices:
        indice_central = diccionario_palabras[corpus[i]]
        contexto_indices = [diccionario_palabras[corpus[i + j]] for j in indices_contexto_offset]

        excluidos = set(contexto_indices + [indice_central])
        posibles_negativos = list(vocab_indices - excluidos)
        negativos_elegidos = random.sample(posibles_negativos, min(negativos, len(posibles_negativos)))

        indices_tuplas.append((indice_central, contexto_indices, negativos_elegidos))

    # Entrenamiento
    print("Inicia entrenamiento")
    for epoca in range(epocas):
        for i, (indice_central, contexto_indices, negativos_indices) in enumerate(indices_tuplas):
            h = W[indice_central].reshape(-1, 1)
            subconjunto = contexto_indices + negativos_indices

            # Propagación
            u_sub = W_prima[:, subconjunto].T @ h
            y = 1 / (1 + cp.exp(-u_sub))  # sigmoid

            # Actualización
            indices_positivos = [subconjunto.index(idx) for idx in contexto_indices if idx in subconjunto]
            EL_sub = y
            EL_sub[indices_positivos] -= 1

            W_prima[:, subconjunto] -= n * (h @ EL_sub.T)
            EH = W_prima[:, subconjunto] @ EL_sub
            W[indice_central] -= n * EH.T[0]

            if i % 1000 == 0:
                print(f"termino palabra: {i}, epoca:{epoca}")

        print(f"termino epoca: {epoca}")
        if epoca % 50 == 0:
            nombre_archivo = f'pesos_skipgram_{nombre_pc}_epoca{epoca}_neuronas{neuronas_oculta}.npz'
            cp.savez(nombre_archivo, W1=W, W2=W_prima, eta=n, N=neuronas_oculta, C=len(contexto)/2)
            print(f"Pesos e hiperparámetros guardados exitosamente en '{nombre_archivo}'")

    return W, W_prima

def obtener_negativas_por_contexto(corpus, indice, contexto, num_negativas=5):
    inicio = max(0, indice - contexto)
    fin = min(len(corpus), indice + contexto + 1)

    contexto_palabras = [corpus[i] for i in range(inicio, fin) if i != indice]

    # Si hay menos de las que pedimos, se rellena desde los bordes
    if len(contexto_palabras) < num_negativas:
        faltan = num_negativas - len(contexto_palabras)
        # Tomar palabras extra empezando desde los bordes del corpus
        extra = [corpus[i] for i in range(len(corpus)) if i not in range(inicio, fin)]
        contexto_palabras.extend(extra[:faltan])

    return [diccionario_palabras[p] for p in contexto_palabras[:num_negativas]]





def skip_gram_negativos_indices_sinaleatorios(diccionario_palabras, corpus,nombre_pc, neuronas_oculta, n, contexto, epocas,negativos, W=None, W_prima=None):

    cardinal_V = len(diccionario_palabras)
    if W is None and W_prima is None:

        W = cp.random.uniform(0,1,(cardinal_V, neuronas_oculta))
        W_prima = cp.random.uniform(0,1,(neuronas_oculta, cardinal_V))

    else:
        W = cp.asarray(W)
        W_prima = cp.asarray(W_prima)

    indices = [i for i in range(contexto,(len(corpus)-contexto))]
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]

    indices_tuplas = [
    (
        diccionario_palabras[corpus[i]],
        [
            diccionario_palabras[corpus[i + j]]  
            for j in indices_contexto
        ],
        obtener_negativas_por_contexto(corpus,i,contexto,negativos)
    )
    for i in indices
]

    for epoca in range(epocas):

        for i, (indice_central, indices_contexto_local, negativos) in enumerate(indices_tuplas):
            print("comenzo")

            h = W[indice_central].reshape(-1,1)
            subconjunto = indices_contexto_local + negativos  # usar la variable local correcta
            u_sub = W_prima[:, subconjunto].T @ h
            y = sigmoid(u_sub)
            subconjunto_positivos = [subconjunto.index(idx) for idx in indices_contexto_local if idx in subconjunto]
            EL_sub = y
            EL_sub[subconjunto_positivos] -= 1
            W_prima[:, subconjunto] -= n*(h@EL_sub.T)
            EH = W_prima[:, subconjunto]@EL_sub
            W[indice_central] -= n*EH.T

            print("termino")
            if i % 1000 == 0:
                print(f"termino palabra: {i}, epoca:{epoca}")
  
        print(f"termino epoca: {epoca}")
        if i % 50 == 0:
            nombre_archivo = f'pesos_cbow_{nombre_pc}_epoca{epoca}.npz'
            #W1_np = cp.asnumpy(W)
            #W2_np = cp.asnumpy(W_prima)
            cp.savez(nombre_archivo, W1=W, W2=W_prima, eta=n, N=neuronas_oculta, C=contexto)
            print(f"Pesos e hiperparámetros guardados exitosamente en '{nombre_archivo}'")
    return W, W_prima

skip_gram_negativos_indices(diccionario_palabras, words,"nico_pc", 90, 0.01, 3, 1001,5, W=None, W_prima=None)
skip_gram_negativos_indices(diccionario_palabras, words,"nico_pc", 40, 0.01, 3, 1001,5, W=None, W_prima=None)
skip_gram_negativos_indices(diccionario_palabras, words,"nico_pc", 65, 0.01, 3, 1001,5, W=None, W_prima=None)