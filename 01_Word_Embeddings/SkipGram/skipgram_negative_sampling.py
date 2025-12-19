import random
import numpy as cp
from auxiliares_skipgram import palabras_a_indice,sigmoid,words,generar_tuplas_con_negativos,generar_tuplas_con_negativos_random


def skip_gram_negativos(palabras_a_indice, corpus, neuronas_oculta, n, contexto, epocas, negativos, W=None, W_prima=None):

    cardinal_V = len(palabras_a_indice)
    if W is None and W_prima is None:
        W = cp.random.uniform(0, 1, (cardinal_V, neuronas_oculta))
        W_prima = cp.random.uniform(0, 1, (neuronas_oculta, cardinal_V))
    else:
        W = cp.asarray(W)
        W_prima = cp.asarray(W_prima)

    # Construir tuplas de entrenamiento (central, contexto, negativos)
    indices_tuplas = generar_tuplas_con_negativos(corpus, palabras_a_indice, contexto, negativos)

    print("Inicia entrenamiento")
    for epoca in range(epocas):
        for i, (indice_central, contexto_indices, negativos_indices) in enumerate(indices_tuplas):
            
            # dimension de h (neuronas_oculta, 1)
            h = W[indice_central].reshape(-1, 1)

            subconjunto = contexto_indices + negativos_indices

            # Propagación

            
            # W_prima es (neuronas_oculta, cardinal_V) y W_prima.T es (cardinal_V, neuronas_oculta)
            #W_prima[:, subconjunto] es (neuronas_oculta, len(subconjunto))
            #dimension (len(subconjunto), 1) = (len(subconjunto), neuronas_oculta)@(neuronas_oculta, 1)
            u_sub = W_prima[:, subconjunto].T @ h
            y = sigmoid(u_sub)

            # Actualización

            ## Recupero los índices de los contextos positivos dentro del subconjunto,
            #  ya que el subconjunto incluye negativos también y cambio el orden
            indices_positivos = [subconjunto.index(idx) for idx in contexto_indices if idx in subconjunto]

            EL_sub = y


            EL_sub[indices_positivos] -= 1

            # EL_sub es (len(subconjunto), 1), EL_sub.T es (1, len(subconjunto))
            # (neuronas_oculta, len(subconjunto)) = (neuronas_oculta, 1)@(1, len(subconjunto))
            W_prima[:, subconjunto] -= n * (h @ EL_sub.T)


            # (neuronas_oculta, 1) = (neuronas_oculta, len(subconjunto))@(len(subconjunto), 1)
            EH = W_prima[:, subconjunto] @ EL_sub


            # (1, neuronas_oculta) = (1, neuronas_oculta) - (1, neuronas_oculta), porque w[indice_central] esta en (1, neuronas_oculta)
            W[indice_central] -= n * EH.T[0]

            if i % 1000 == 0:
                print(f"termino palabra: {i}, epoca:{epoca}")

        print(f"termino epoca: {epoca}")
        if epoca % 50 == 0:
            nombre_archivo = f'pesos_skipgram_epoca{epoca}_neuronas{neuronas_oculta}.npz'
            #W1_np = cp.asnumpy(W)
            #W2_np = cp.asnumpy(W_prima)
            cp.savez(nombre_archivo, W1=W, W2=W_prima, eta=n, N=neuronas_oculta, C=len(contexto)/2)
            print(f"Pesos e hiperparámetros guardados exitosamente en '{nombre_archivo}'")

    return W, W_prima




def skip_gram_negativos_aleatorios(diccionario_palabras, corpus,nombre_pc, neuronas_oculta, n, contexto, epocas,negativos, W=None, W_prima=None):

    cardinal_V = len(palabras_a_indice)
    if W is None and W_prima is None:
        W = cp.random.uniform(0, 1, (cardinal_V, neuronas_oculta))
        W_prima = cp.random.uniform(0, 1, (neuronas_oculta, cardinal_V))
    else:
        W = cp.asarray(W)
        W_prima = cp.asarray(W_prima)

    # Construir tuplas de entrenamiento (central, contexto, negativos)
    indices_tuplas = generar_tuplas_con_negativos_random(corpus, palabras_a_indice, contexto, negativos)

    print("Inicia entrenamiento")
    for epoca in range(epocas):
        for i, (indice_central, contexto_indices, negativos_indices) in enumerate(indices_tuplas):

            h = W[indice_central].reshape(-1, 1)

            subconjunto = contexto_indices + negativos_indices

            # Propagación

            u_sub = W_prima[:, subconjunto].T @ h
            y = sigmoid(u_sub)

            # Actualización
            ## Recupero los índices de los contextos positivos dentro del subconjunto, ya que el subconjunto incluye negativos también y cambio el orden
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
            nombre_archivo = f'pesos_skipgram_epoca{epoca}_neuronas{neuronas_oculta}.npz'
            #W1_np = cp.asnumpy(W)
            #W2_np = cp.asnumpy(W_prima)
            cp.savez(nombre_archivo, W1=W, W2=W_prima, eta=n, N=neuronas_oculta, C=len(contexto)/2)
            print(f"Pesos e hiperparámetros guardados exitosamente en '{nombre_archivo}'")

    return W, W_prima