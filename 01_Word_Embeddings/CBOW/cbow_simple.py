import numpy as np
from auxiliares_cbow import softmax, palabras_a_indice,corpus_modificado,generar_tuplas_nuevo,indices_a_palabras, cargar_modelo_completo
#import cupy as cp


def CBOW(palabras_a_indice, corpus, neuronas_oculta, W=None, W_prima=None, contexto=5, epocas=1000, n=0.01):
    
    cardinal_V = len(palabras_a_indice)

    if  W_prima is None and W is None:
        W = np.random.normal(0,1,(cardinal_V, neuronas_oculta))
        W_prima = np.random.normal(0,1,(neuronas_oculta, cardinal_V))
    else:
        W = np.asarray(W)
        W_prima = np.asarray(W_prima)

    indices_tuplas = generar_tuplas_nuevo(corpus, palabras_a_indice, contexto)

    
    for epoca in range(epocas):

        E = 0
        E_estrella = 0
        
        for i, (indice_central, indices_contextos) in enumerate(indices_tuplas):

            h = np.mean(W[indices_contextos], axis=0).reshape(-1,1)

            u = W_prima.T@h

            y, sum_exp = softmax(u)
            #print(u.shape)  # Asegúrate de que `u` tiene el tamaño esperado
            #print(indice_central)  # Verifica el índice central que estás usando
            E += (np.log(sum_exp) - u[indice_central])
            
            E_estrella += (1 - y[indice_central])

            e = y.copy()

            e[indice_central] -= 1

            EH = W_prima@e

            W_prima -= n*(h@e.T)

            

            W[indices_contextos] -=n * EH.T / len(indices_contextos)

            if i % 100000 == 0:
                print(f"Termino palabra: {i}, epoca:{epoca}")

        print(f"Termino epoca: {epoca} con E {E} y E* {E_estrella}")
        if epoca % 50 == 0:
            nombre_archivo = f'pesos_cbow_mejores_epoca{epoca}_neurona_oculta{neuronas_oculta}.npz'
            W1_np = np.asnumpy(W)
            W2_np = np.asnumpy(W_prima)
            np.savez(nombre_archivo, W1=W1_np, W2=W2_np, eta=n, N=neuronas_oculta, C=contexto)
            print(f"Pesos e hiperparámetros guardados exitosamente en '{nombre_archivo}'")
    return W, W_prima

W, W_prima = CBOW(palabras_a_indice, corpus_modificado, 130, contexto=5, epocas=2000, n=0.01)





