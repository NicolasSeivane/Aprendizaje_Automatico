import numpy as np
import random
import cupy as cp

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

            W[indice_central] -= n * EH.T
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

            W[indice] -= n * EH.T
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def skip_gram_negativos_indices(diccionario_palabras, corpus,nombre_pc, neuronas_oculta, n, contexto, epocas,negativos, W=None, W_prima=None):

    cardinal_V = len(diccionario_palabras)
    if W is None and W_prima is None:

        W = np.random.uniform(0,1,(cardinal_V, neuronas_oculta))
        W_prima = np.random.uniform(0,1,(neuronas_oculta, cardinal_V))

    else:
        W = cp.asarray(W)
        W_prima = cp.asarray(W_prima)

    indices = [i for i in range(contexto,(len(corpus)-contexto))]
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]
    vocab_indices = list(range(len(diccionario_palabras)))

    indices_tuplas = [
    (
        diccionario_palabras[corpus[i]],
        [
            diccionario_palabras[corpus[i + j]]  
            for j in indices_contexto
        ],
        random.sample(
            list(set(vocab_indices) - set([diccionario_palabras[corpus[i + j]] for j in indices_contexto])),
            k=negativos 
        )
    )
    for i in indices
]

    for epoca in range(epocas):

        for i, (indice_central,contexto, negativos)in enumerate(indices_tuplas):

            h = W[indice_central].reshape(-1,1)

            subconjunto = indices_contexto + negativos  # lista de índices

            u_sub = W_prima[:, subconjunto].T @ h

            y = sigmoid(u_sub)

            subconjunto_positivos = [subconjunto.index(indice) for indice in indices_contexto if indice in subconjunto]

            EL_sub = y

            EL_sub[subconjunto_positivos] -= 1


            W_prima[:, subconjunto] -= n*(h@EL_sub.T)

            EH = W_prima[:, subconjunto]@EL_sub

            W[indice_central] -= n*EH.T

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

def obtener_negativas_por_contexto(corpus, indice, contexto, num_negativas=5):

    antes_contexto = range(indice - contexto, indice)
    despues_contexto = range(indice + 1, indice + contexto + 1)
    
    palabras_antes = corpus[antes_contexto] if indice - contexto >= 0 else [] # EVITO INDICE 0
    palabras_despues = corpus[despues_contexto] if indice + contexto < len(corpus) else [] # evito salirme del limite corpus
    
    num_negativas_antes = min(num_negativas // 2, len(palabras_antes))
    num_negativas_despues = min(num_negativas // 2, len(palabras_despues))

    negativas = []

    negativas.extend(palabras_antes[:num_negativas_antes])

    negativas.extend(palabras_despues[:num_negativas_despues])

    if len(negativas) < num_negativas:
     
        faltantes = num_negativas - len(negativas)

        if len(palabras_antes) < num_negativas // 2:
            negativas.extend(palabras_despues[num_negativas_despues:num_negativas_despues + faltantes])

        elif len(palabras_despues) < num_negativas // 2:
            negativas.extend(palabras_antes[num_negativas_antes:num_negativas_antes + faltantes])

    if len(negativas) < num_negativas:
        palabras_restantes = [palabra for i, palabra in enumerate(corpus) if i != indice]
        negativas.extend(random.sample(palabras_restantes, num_negativas - len(negativas)))

    return negativas





def skip_gram_negativos_indices_sinaleatorios(diccionario_palabras, corpus,nombre_pc, neuronas_oculta, n, contexto, epocas,negativos, W=None, W_prima=None):

    cardinal_V = len(diccionario_palabras)
    if W is None and W_prima is None:

        W = np.random.uniform(0,1,(cardinal_V, neuronas_oculta))
        W_prima = np.random.uniform(0,1,(neuronas_oculta, cardinal_V))

    else:
        W = cp.asarray(W)
        W_prima = cp.asarray(W_prima)

    indices = [i for i in range(contexto,(len(corpus)-contexto))]
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]
    vocab_indices = list(range(len(diccionario_palabras)))

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

        for i, (indice_central,contexto, negativos)in enumerate(indices_tuplas):

            h = W[indice_central].reshape(-1,1)

            subconjunto = indices_contexto + negativos  # lista de índices

            u_sub = W_prima[:, subconjunto].T @ h

            y = sigmoid(u_sub)

            subconjunto_positivos = [subconjunto.index(indice) for indice in indices_contexto if indice in subconjunto]

            EL_sub = y

            EL_sub[subconjunto_positivos] -= 1


            W_prima[:, subconjunto] -= n*(h@EL_sub.T)

            EH = W_prima[:, subconjunto]@EL_sub

            W[indice_central] -= n*EH.T

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
