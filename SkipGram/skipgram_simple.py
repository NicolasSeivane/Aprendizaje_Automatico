import random
import numpy as cp
from auxiliares_skipgram import palabras_a_indice,softmax,words,generar_tuplas

def skip_gram(palabras_a_indice, corpus, neuronas_oculta, n, contexto, epocas, W=None, W_prima=None):
    cardinal_V = len(palabras_a_indice)

    if W is None and W_prima is None:
        W = cp.random.normal(0,1,(cardinal_V, neuronas_oculta))
        W_prima = cp.random.normal(0,1,(neuronas_oculta, cardinal_V))
    else:
        W = cp.asarray(W)
        W_prima = cp.asarray(W_prima)


    indices_tuplas = generar_tuplas(corpus, palabras_a_indice, contexto)


    for epoca in range(epocas):

        for i, (indice_central,indices_contexto)in enumerate(indices_tuplas):

            ## dimension de h (neuronas_oculta, 1)
            h = W[indice_central].reshape(-1,1)


            ## (cardinal_V, 1) = (cardinal_V, neuronas_oculta)@(neuronas_oculta, 1)
            u = W_prima.T@h

            y = softmax(u)

            EL = y

            EL[indices_contexto] -= 1

            ## (neuronas_oculta, cardinal_V) = (neuronas_oculta, 1)@(1, cardinal_V)
            W_prima -= n * (h @ EL.T)
            
            ## (neuronas_oculta, 1) = (neuronas_oculta, cardinal_V)@(cardinal_V, 1)
            EH = W_prima @ EL

            ## (1, neuronas_oculta) = (1, neuronas_oculta) - (1, neuronas_oculta), porque w[indice_central] esta en (1, neuronas_oculta)
            W[indice_central] -= n * EH.T[0]
            if i % 1000 == 0:
                print(f"termino palabra: {i}, epoca:{epoca}")

        print(f"termino epoca: {epoca}")

        if i % 50 == 0:
            nombre_archivo = f'pesos_skipgram_epoca{epoca}_neuronas{neuronas_oculta}.npz'
            #W1_np = cp.asnumpy(W)
            #W2_np = cp.asnumpy(W_prima)
            cp.savez(nombre_archivo, W1=W, W2=W_prima, eta=n, N=neuronas_oculta, C=len(contexto)/2)
            print(f"Pesos e hiperparámetros guardados exitosamente en '{nombre_archivo}'")
    return W, W_prima

skip_gram(palabras_a_indice, words, 175, 0.01, 7, 1000)