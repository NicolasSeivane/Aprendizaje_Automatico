import numpy as np

def cargar_modelo_completo(nombre_archivo='pesos_cbow_pc2_epoca0.npz'):
    """
    Carga los pesos W1, W2 y los hiperparámetros N, C y eta
    desde un archivo .npz.
    """
    try:
        data = np.load(nombre_archivo)

        W1 = data['W1']
        W2 = data['W2']

        N = data['N'].item()
        C = data['C'].item()
        eta = data['eta'].item()

        print()

        return W1, W2, N, C, eta

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo}'.")
        return None, None, None, None, None
    
def generar_ventana(corpus, palabras_a_indice, contexto, indices_a_embeddings):
    indices = range(contexto, len(corpus))
    indices_contexto = range(-contexto, 0)

    contexto_a_central = {}
    indice_a_palabra = {v: k for k, v in palabras_a_indice.items()}
    X = []
    Y = []
    Y2 = []
    for i in indices:
        palabra_central = palabras_a_indice[corpus[i]]
        contexto_actual = tuple(palabras_a_indice[corpus[i+j]] for j in indices_contexto)

        # Si ya existe el mismo contexto pero con otra palabra central
        if contexto_actual in contexto_a_central:
            if contexto_a_central[contexto_actual] != palabra_central:
                continue
        else:
            contexto_a_central[contexto_actual] = palabra_central
        
        ventana = np.concatenate([indices_a_embeddings[idx] for idx in contexto_actual], axis=0)
        X.append(ventana.flatten())          # Aplanamos para que quede 1D
        Y.append(indices_a_embeddings[palabra_central])
        Y2.append(palabra_central)
    return np.array(X), np.array(Y), np.array(Y2, dtype=np.int32)


with open("C:\\Users\\User\\Documents\\GitHub\\Aprendizaje_Automatico\\Evaluacion_Modelos\\corpus_junto2_todos_los_fuegos.txt", "r", encoding="utf-8") as f:
    words = f.read().splitlines()

corpus_modificado = words.copy()

palabras_a_indice = {}
indices_a_palabras = {}
diccionario_onehot = {}
diccionario_onehot_a_palabra = {}
diccionario_conteo = {}
indices_a_embeddings = {}

W1, W2,N, C, eta = cargar_modelo_completo("C:\\Users\\User\\Documents\\GitHub\\Aprendizaje_Automatico\\pesos_cbow_neg_epoca1500_contexto_5.npz")


for token in words:
    if token not in palabras_a_indice:
        index = len(palabras_a_indice)
        palabras_a_indice[token] = index
        indices_a_palabras[index] = token
        indices_a_embeddings[index] = W1[index].reshape(1,-1)
        diccionario_conteo[token] = 1 
    else:
        diccionario_conteo[token] += 1 


cardinal_V = len(palabras_a_indice)

for token, idx in list(palabras_a_indice.items()):

    one_hot_vector = np.zeros(cardinal_V)
    one_hot_vector[idx] = 1
    diccionario_onehot[token] = one_hot_vector
    diccionario_onehot_a_palabra[str(one_hot_vector)] = token

def softmax(x):
    x = np.asarray(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predecir(ventana, modelo,topk=5):

    prediccion = modelo.predict(ventana, verbose=0)
    prediccion = prediccion.flatten() 
    candidatos = np.argsort(-prediccion)
    palabra_top1 = indices_a_palabras[candidatos[0]]
    palabras_topk = [indices_a_palabras[i] for i in candidatos[:topk]]

    return palabra_top1, palabras_topk


def predecir_cbow_completo_one_hot(palabras_a_indice, corpus, modelo, indices_a_embeddings, topk=5):

    x_gen,y1,y2_one_hot = generar_ventana(corpus, palabras_a_indice, 10, indices_a_embeddings)
    aciertos_top1 = 0
    aciertos_topk = 0
    palabras_mal_predichas = []

    totales = len(x_gen)

    for i in range(x_gen.shape[0]):
        palabra_top1, palabras_topk = predecir(x_gen[i].reshape(1, -1), modelo, topk=topk)

        objetivo = np.argmax(y2_one_hot[i].flatten())

        palabra_objetivo = indices_a_palabras[objetivo]
        # Top-1
        if palabra_objetivo == palabra_top1:
            aciertos_top1 += 1

        # Top-k
        if palabra_objetivo in palabras_topk:
            aciertos_topk += 1
        else:
            palabras_mal_predichas.append(palabra_objetivo)

        # Progreso
        if i % 100 == 0 and i > 0:
            progreso = (i / totales) * 100
            print(f"Progreso: [{progreso:.1f}%] - Top1: {aciertos_top1/totales:.4f}%, Top{topk}: {aciertos_topk/totales:.4f}%", end='\r')

    accuracy_top1 = aciertos_top1 / totales
    accuracy_topk = aciertos_topk / totales

    print(f"\nResultados completos: Top1={accuracy_top1*100:.2f}%, Top{topk}={accuracy_topk*100:.2f}%")

    return accuracy_top1, accuracy_topk, palabras_mal_predichas