import numpy as np
import random
from auxiliares_cbow import sigmoid, palabras_a_indice,words,generar_tuplas_con_negativos,generar_tuplas_con_negativos_random
import cupy as cp

def cbow_negativos_cercanos(palabras_a_indice, corpus, neuronas_oculta, n, contexto, epocas, negativos,
                            W=None, W_prima=None):

    cardinal_V = len(palabras_a_indice)

    if W is None and W_prima is None:
        W = cp.random.uniform(0, 1, (cardinal_V, neuronas_oculta))
        W_prima = cp.random.uniform(0, 1, (neuronas_oculta, cardinal_V))
    else:
        W = cp.asarray(W)
        W_prima = cp.asarray(W_prima)

    # Cada tupla = (target, contexto, negativos)
    indices_tuplas = generar_tuplas_con_negativos(corpus, palabras_a_indice, contexto, negativos)

    for epoca in range(epocas):

        for i, (indice_central, contexto_idx, negativos_idx) in enumerate(indices_tuplas):

            h = cp.mean(W[contexto_idx], axis=0).reshape(-1, 1)

            subconjunto = list(set([indice_central] + negativos_idx))
            u_sub = W_prima[:, subconjunto].T @ h  # (len(subconjunto), 1)

            y = sigmoid(u_sub)

            EL_sub = y


            pos_idx = subconjunto.index(indice_central)


            EL_sub[pos_idx] -= 1


            W_prima[:, subconjunto] -= n * (h @ EL_sub.T)


            EH = W_prima[:, subconjunto] @ EL_sub  # (dim, 1)
            
            W[contexto_idx] -= n * EH.T / len(contexto_idx)

            if i % 1000 == 0:
                print(f"Termino palabra: {i}, epoca:{epoca}")

        print(f"Termino epoca: {epoca}")

        # Guardado periódico
        if epoca % 50 == 0:

            nombre_archivo = f'pesos_cbow_neg_epoca{epoca}.npz'

            W1_np = cp.asnumpy(W)
            W2_np = cp.asnumpy(W_prima)
            np.savez(nombre_archivo, W1=W1_np, W2=W2_np,
                     eta=n, N=neuronas_oculta, C=contexto, num_neg=negativos)
            print(f"Pesos guardados en '{nombre_archivo}'")

    return W, W_prima


def cbow_negativos_random(palabras_a_indice, corpus, neuronas_oculta, n, contexto, epocas, negativos,
                            W=None, W_prima=None):

    cardinal_V = len(palabras_a_indice)

    if W is None and W_prima is None:
        W = cp.random.uniform(0, 1, (cardinal_V, neuronas_oculta))
        W_prima = cp.random.uniform(0, 1, (neuronas_oculta, cardinal_V))
    else:
        W = cp.asarray(W)
        W_prima = cp.asarray(W_prima)

    # Cada tupla = (target, contexto, negativos)
    indices_tuplas = generar_tuplas_con_negativos_random(corpus, palabras_a_indice, contexto, negativos)

    for epoca in range(epocas):

        for i, (indice_central, contexto_idx, negativos_idx) in enumerate(indices_tuplas):

            h = cp.mean(W[contexto_idx], axis=0).reshape(-1, 1)

            subconjunto = list(set([indice_central] + negativos_idx))
            u_sub = W_prima[:, subconjunto].T @ h  # (len(subconjunto), 1)

            y = sigmoid(u_sub)

            EL_sub = y


            pos_idx = subconjunto.index(indice_central)


            EL_sub[pos_idx] -= 1


            W_prima[:, subconjunto] -= n * (h @ EL_sub.T)


            EH = W_prima[:, subconjunto] @ EL_sub  # (dim, 1)
            
            W[contexto_idx] -= n * EH.T / len(contexto_idx)

            if i % 1000 == 0:
                print(f"Termino palabra: {i}, epoca:{epoca}")

        print(f"Termino epoca: {epoca}")

        # Guardado periódico
        if epoca % 50 == 0:

            nombre_archivo = f'pesos_cbow_neg_epoca{epoca}.npz'

            W1_np = cp.asnumpy(W)
            W2_np = cp.asnumpy(W_prima)
            np.savez(nombre_archivo, W1=W1_np, W2=W2_np,
                     eta=n, N=neuronas_oculta, C=contexto, num_neg=negativos)
            print(f"Pesos guardados en '{nombre_archivo}'")

    return W, W_prima

cbow_negativos_cercanos(palabras_a_indice, words, 55, n=0.05, contexto=2, epocas=200, negativos=5)

cbow_negativos_random(palabras_a_indice, words, 55, n=0.05, contexto=2, epocas=200, negativos=5)