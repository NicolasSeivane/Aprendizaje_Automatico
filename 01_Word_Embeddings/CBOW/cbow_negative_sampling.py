import numpy as np
import random
import os
from auxiliares_cbow import sigmoid, palabras_a_indice,generar_tuplas_nuevo,words, generar_negativos_pool,generar_distribucion_negativa,cargar_modelo_completo
#import cupy as cp

def cbow_negativos_cercanos(palabras_a_indice, corpus,neuronas_oculta, n, contexto, epocas, negativos,
                            W=None, W_prima=None):

    cardinal_V = len(palabras_a_indice)

    if W is None and W_prima is None:
        W = np.random.uniform(0, 1, (cardinal_V, neuronas_oculta))
        W_prima = np.random.uniform(0, 1, (neuronas_oculta, cardinal_V))
    else:
        W = np.asarray(W)
        W_prima = np.asarray(W_prima)

    # Cada tupla = (target, contexto, negativos)
    indices_tuplas = generar_tuplas_nuevo(corpus, palabras_a_indice, contexto)

    distribucion = generar_distribucion_negativa(corpus)
    vocab_palabras = list(distribucion.keys())
    vocab_indices = np.array([palabras_a_indice[p] for p in vocab_palabras])
    probs = np.array([distribucion[p] for p in vocab_palabras])
    probs = probs / probs.sum()  # Normalizar

    for epoca in range(epocas):

        negativos_tuplas = generar_negativos_pool(indices_tuplas, vocab_indices, probs, negativos)

        if epoca == 0:
            print("Arranco Loco")

        E_estrella = 0

        for i, (indice_central, contexto_idx, negativos_idx) in enumerate(negativos_tuplas):

            h = np.mean(W[contexto_idx], axis= 0).reshape(-1, 1)

            subconjunto = list(set([indice_central] + negativos_idx))

            u_sub = W_prima[:, subconjunto].T @ h  # (len(subconjunto), 1)

            y = sigmoid(u_sub)

            

            EL_sub = y


            pos_idx = subconjunto.index(indice_central)

            E_estrella += (1 - y[pos_idx])

            avg_loss = E_estrella / len(indices_tuplas)

            EL_sub[pos_idx] -= 1

            EH = W_prima[:, subconjunto] @ EL_sub  # (dim, 1)

            W_prima[:, subconjunto] -= n * (h @ EL_sub.T)

            W[contexto_idx] -= n * EH.T / len(contexto_idx)

            if i % 100000 == 0:

                print(f"Termino palabra: {i}, epoca:{epoca}")

        print(f"Termino epoca: {epoca} con E* {E_estrella}, promedio de loss: {avg_loss}")

        # Guardado periódico
        if epoca % 50 == 0:

            nombre_archivo = os.path.join("..", "..", "models", f'pesos_cbow_neg_epoca{epoca}_contexto_{contexto}.npz')

            #W1_np = cp.asnumpy(W)
            #W2_np = cp.asnumpy(W_prima)
            np.savez(nombre_archivo, W1=W, W2=W_prima,
                     eta=n, N=neuronas_oculta, C=contexto, num_neg=negativos)
            print(f"Pesos guardados en '{nombre_archivo}'")

    return W, W_prima


MODEL_PATH = os.path.join("..", "..", "models", "pesos_cbow_neg_epoca200_contexto_5.npz")
W1, W2, N, C, eta = cargar_modelo_completo(MODEL_PATH)


cbow_negativos_cercanos(palabras_a_indice=palabras_a_indice,W=W1,W_prima=W2, corpus=words, n=0.01, contexto=5, epocas=2000, negativos=15,neuronas_oculta=130)

