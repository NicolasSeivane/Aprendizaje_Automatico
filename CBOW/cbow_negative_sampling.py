import numpy as np
import random
from auxiliares_cbow import sigmoid, palabras_a_indice,generar_tuplas_nuevo_negativos,corpus_modificado, cargar_modelo_completo
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
    indices_tuplas = generar_tuplas_nuevo_negativos(corpus, palabras_a_indice, contexto)


   


    for epoca in range(epocas):

        E_estrella = 0

        for i, (indice_central, contexto_idx, negativos_idx) in enumerate(indices_tuplas):

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

            nombre_archivo = f'pesos_cbow_neg_epoca{epoca}_contexto_{contexto}.npz'

            #W1_np = cp.asnumpy(W)
            #W2_np = cp.asnumpy(W_prima)
            np.savez(nombre_archivo, W1=W, W2=W_prima,
                     eta=n, N=neuronas_oculta, C=contexto, num_neg=negativos)
            print(f"Pesos guardados en '{nombre_archivo}'")

    return W, W_prima


#W1, W2, N, C, eta = cargar_modelo_completo("C:\\Users\\User\\Documents\\GitHub\\Aprendizaje_Automatico\\pesos_cbow_neg_epoca150_neuronas_110.npz")


cbow_negativos_cercanos(palabras_a_indice=palabras_a_indice, corpus=corpus_modificado, n=0.05, contexto=5, epocas=2000, negativos=15,neuronas_oculta=130)

