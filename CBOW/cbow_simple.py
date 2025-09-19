import numpy as np
import random
from auxiliares_cbow import softmax, palabras_a_indice,words,generar_tuplas
import cupy as cp


def CBOW(palabras_a_indice, corpus, neuronas_oculta, W=None, W_prima=None, contexto=5, epocas=1000, n=0.01):
    
    cardinal_V = len(palabras_a_indice)

    if not W_prima and not W:
        W = cp.random.normal(0,1,(cardinal_V, neuronas_oculta))
        W_prima = cp.random.normal(0,1,(neuronas_oculta, cardinal_V))
    else:
        W = cp.asarray(W)
        W_prima = cp.asarray(W_prima)

    indices_tuplas = generar_tuplas(corpus, palabras_a_indice, contexto)

    for epoca in range(epocas):

        for i, (indice_central, indices_contextos) in enumerate(indices_tuplas):

            h = cp.mean(W[indices_contextos], axis=0).reshape(-1,1)

            u = W_prima.T@h

            y = softmax(u)

            e = y

            e[indice_central] -= 1

            W_prima -= n*(h@e.T)

            EH = W_prima@e

            W[indices_contextos] -=n * EH.T / len(indices_contextos)

            if i % 100000 == 0:
                print(f"Termino palabra: {i}, epoca:{epoca}")

        print(f"Termino epoca: {epoca}")
        if epoca % 50 == 0:
            nombre_archivo = f'pesos_cbow_epoca{epoca}_neurona_oculta{neuronas_oculta}.npz'
            W1_np = cp.asnumpy(W)
            W2_np = cp.asnumpy(W_prima)
            np.savez(nombre_archivo, W1=W1_np, W2=W2_np, eta=n, N=neuronas_oculta, C=contexto)
            print(f"Pesos e hiperparámetros guardados exitosamente en '{nombre_archivo}'")
    return W, W_prima

W, W_prima = CBOW(palabras_a_indice, words, 100, contexto=2, epocas=200, n=0.05)






