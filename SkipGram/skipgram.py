import numpy as np
import random

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


def skip_gram(diccionario_palabras, corpus, neuronas_oculta, n, contexto, epocas, W=None, W_prima=None):
    cardinal_V = len(diccionario_palabras)
    if W is None and W_prima is None:

        W = np.random.uniform(0,1,(cardinal_V, neuronas_oculta))
        W_prima = np.random.uniform(0,1,(neuronas_oculta, cardinal_V))

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



def skip_gram_indices(diccionario_palabras, corpus, neuronas_oculta, n, contexto, epocas, W=None, W_prima=None):
    cardinal_V = len(diccionario_palabras)
    if W is None and W_prima is None:
        
        W = np.random.uniform(0,1,(cardinal_V, neuronas_oculta))
        W_prima = np.random.uniform(0,1,(neuronas_oculta, cardinal_V))

    indices = [i for i in range(contexto,(len(corpus)-contexto))]
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]
    indices_tuplas = [(i, [i+j for j in indices_contexto]) for i in indices]

    for i in range(epocas):

        for indice, contexto in indices_tuplas:
            
            indice_central = diccionario_palabras[corpus[indice]]

            indices_contexto = [diccionario_palabras[corpus[j]] for j in contexto]

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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def skip_gram_negativos(diccionario_palabras, corpus, neuronas_oculta, n, contexto, epocas,negativos, W=None, W_prima=None):
    cardinal_V = len(diccionario_palabras)
    if W is None and W_prima is None:

        W = np.random.uniform(0,1,(cardinal_V, neuronas_oculta))
        W_prima = np.random.uniform(0,1,(neuronas_oculta, cardinal_V))

    cardinal_corpus = len(corpus)

    for i in range(epocas):

        for indice in range(contexto-1,(cardinal_corpus-contexto)+1):
            
            palabra = corpus[indice]
            indice_central = diccionario_palabras[palabra]

            palabras_contexto = corpus[indice-contexto:indice] + corpus[indice+1:indice+contexto+1]
            indices_contexto = [diccionario_palabras[palabra] for palabra in palabras_contexto]


            palabras_negativos = random.sample([palabra for palabra in corpus if palabra not in palabras_contexto+[palabra]], negativos)
            indices_negativos = [diccionario_palabras[palabra] for palabra in palabras_negativos]

            #indices_negativos = generar_negativos(indice,indices_contexto, negativos)

            h = W[indice_central].reshape(-1,1)

            subconjunto = indices_contexto + indices_negativos  # lista de índices

            u_sub = W_prima[:, subconjunto].T @ h

            y = sigmoid(u_sub)

            subconjunto_positivos = [subconjunto.index(idx_vocab) for idx_vocab in indices_contexto if idx_vocab in subconjunto]

            EL_sub = y.copy()

            EL_sub[subconjunto_positivos] -= 1


            W_prima[:, subconjunto] = W_prima[:, subconjunto] -n*(h@EL_sub.T)

            EH = W_prima[:, subconjunto]@EL_sub

            W[indice_central] = W[indice_central] -n*EH.T

            print("llegamos")
  
        print(f"termino epoca: {i}")
    return W, W_prima



def skip_gram_negativos_indices(diccionario_palabras, corpus, neuronas_oculta, n, contexto, epocas,negativos, W=None, W_prima=None):

    cardinal_V = len(diccionario_palabras)
    if W is None and W_prima is None:

        W = np.random.uniform(0,1,(cardinal_V, neuronas_oculta))
        W_prima = np.random.uniform(0,1,(neuronas_oculta, cardinal_V))

    indices = [i for i in range(contexto,(len(corpus)-contexto))]
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]
    indices_tuplas = [(i, [i+j for j in indices_contexto]) for i in indices]
    vocab_indices = list(range(len(diccionario_palabras)))

    for i in range(epocas):

        for indice, contexto in indices_tuplas:
            
            indice_central = diccionario_palabras[corpus[indice]]

            indices_contexto = [diccionario_palabras[corpus[j]] for j in contexto]
            
            candidatos = list(set(vocab_indices) - set(indices_contexto + [indice_central]))

            indices_negativos = random.sample(candidatos, negativos)

            h = W[indice_central].reshape(-1,1)

            subconjunto = indices_contexto + indices_negativos  # lista de índices

            u_sub = W_prima[:, subconjunto].T @ h

            y = sigmoid(u_sub)

            subconjunto_positivos = [subconjunto.index(idx_vocab) for idx_vocab in indices_contexto if idx_vocab in subconjunto]

            EL_sub = y.copy()

            EL_sub[subconjunto_positivos] -= 1


            W_prima[:, subconjunto] = W_prima[:, subconjunto] -n*(h@EL_sub.T)

            EH = W_prima[:, subconjunto]@EL_sub

            W[indice_central] = W[indice_central] -n*EH.T

            print("llegamos")
  
        print(f"termino epoca: {i}")
    return W, W_prima
#skip_gram_negativos(diccionario_palabras, words, 200, 0.1, 5, 1000,5)
