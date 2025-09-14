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

def calcular_ej_skipgram(y,indices_contexto):
    e = y.copy()
    for indice in indices_contexto:
        e[indice] -=  1
    return e.reshape(-1,1)
            
    


def denominador_softmax(diccionario_onehot):
    cardinalv = len(diccionario_onehot)
    suma = 0
    for i in range(cardinalv):
        suma += np.exp(diccionario_onehot[i])
    return suma
    
    
    
    
def generar_contexto(diccionario_palabras, indice, contexto, corpus):
    lista = []
    for i in range(-contexto,contexto+1):
        if i != indice:
            palabra = corpus[indice + i]
            lista.append(diccionario_palabras[palabra])
    return lista
    
def softmax2(u):
    y = u.copy()
    max_y = np.max(y) 
    suma = 0
    for i in range(len(u)):
        suma += np.exp( y[i] - max_y)

    for i in range(len(u)):
        y[i] = y[i] / suma
    return y

def softmax(u):
    u_max = np.max(u)
    e_u = np.exp(u - u_max)
    return e_u / e_u.sum()


def skip_gram(diccionario_palabras, corpus, neuronas_oculta, n, contexto, epocas):
    cardinal_V = len(diccionario_palabras)
    W = np.random.uniform(0,1,(cardinal_V, neuronas_oculta))
    W_prima = np.random.uniform(0,1,(neuronas_oculta, cardinal_V))

    cardinal_corpus = len(corpus)

    for i in range(epocas):
        for indice in range(contexto-1,(cardinal_corpus-contexto)+1):
            
            palabra = corpus[indice]
            indice_central = diccionario_palabras[palabra]
            indices_contexto = generar_contexto(diccionario_palabras, indice, contexto, corpus)

            h = W[indice_central].reshape(-1,1)

            u = W_prima.T@h

            y = softmax(u)


            EL = calcular_ej_skipgram(y,indices_contexto)

            W_prima = W_prima -n*(h@EL.T)

            EH = W_prima@EL

            W = W -n*EH.T
            print("LLegamos bien carajo")
  
        print(f"termino epoca: {i}")
    return W, W_prima

skip_gram(diccionario_palabras,words,200,0.01,5,1000)

def generar_negativos(indice,indices_contexto, negativos):
        lista = []
        while len(lista) < negativos:
                neg_idx = np.random.randint(0, cardinal_V)
                if neg_idx not in indices_contexto and neg_idx != indice:
                    lista.append(neg_idx)
        return lista


def calcular_ej_skipgram_negativos(y,indices_contexto,subconjunto):
    e = y.copy()
    for idx_vocab in indices_contexto:
        if idx_vocab in subconjunto:
            idx_sub = subconjunto.index(idx_vocab)
            e[idx_sub] -= 1
    return e

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def skip_gram_negativos(diccionario_palabras, corpus, neuronas_oculta, n, contexto, epocas,negativos):
    cardinal_V = len(diccionario_palabras)
    W = np.random.uniform(0,1,(cardinal_V, neuronas_oculta))
    W_prima = np.random.uniform(0,1,(neuronas_oculta, cardinal_V))

    cardinal_corpus = len(corpus)

    for i in range(epocas):
        for indice in range(contexto-1,(cardinal_corpus-contexto)+1):
            
            palabra = corpus[indice]
            indice_central = diccionario_palabras[palabra]
            indices_contexto = generar_contexto(diccionario_palabras, indice, contexto, corpus)
            indices_negativos = generar_negativos(indice,indices_contexto, negativos)

            h = W[indice_central].reshape(-1,1)

            subconjunto = indices_contexto + indices_negativos  # lista de índices
            u_sub = W_prima[:, subconjunto].T @ h

            y = sigmoid(u_sub)

            EL_sub = calcular_ej_skipgram_negativos(y,indices_contexto,subconjunto)


            W_prima[:, subconjunto] = W_prima[:, subconjunto] -n*(h@EL_sub.T)

            EH = W_prima[:, subconjunto]@EL_sub

            W[indice_central] = W[indice_central] -n*EH.T

            print("llegamos")
  
        print(f"termino epoca: {i}")
    return W, W_prima

#skip_gram_negativos(diccionario_palabras, words, 200, 0.1, 5, 1000,5)
