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

def calcular_ej(y,indice_central):
    e = y.copy()
    e[indice_central] = y[indice_central] - 1
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
    

def calcular_exitacion_bag(indices_contexto,W):
    cantidad = len(indices_contexto)
    suma = 0
    for i in indices_contexto:
        suma += W[i]
    return (suma / cantidad).reshape(-1,1)

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
    e_u = np.exp(u)
    return e_u / e_u.sum()



def bag_of_words(diccionario_palabras, corpus, neuronas_oculta, n, contexto, epocas):
    cardinal_V = len(diccionario_palabras)
    palabras = len(corpus)
    W = np.random.uniform(0,1,(cardinal_V, neuronas_oculta))
    W_prima = np.random.uniform(0,1,(neuronas_oculta, cardinal_V))

    cardinal_corpus = len(corpus)

    for i in range(epocas):
        for indice in range(contexto, (palabras-contexto)):
            
            palabra = corpus[indice]
            indice_central = diccionario_palabras[palabra]
            indices_contexto = generar_contexto(diccionario_palabras, indice, contexto, corpus)

            h = calcular_exitacion_bag(indices_contexto,W)

            u = W_prima.T@h

            y = softmax(u)

            e = calcular_ej(y,indice_central)

            W_prima = W_prima -n*(h@e.T)

            EH = W_prima@e

            W = W -n*(EH.T/len(indices_contexto))
  
        print(f"termino epoca: {i}")
    return W, W_prima

bag_of_words(diccionario_palabras,words,200,0.01,5,1000)

        
    

  


