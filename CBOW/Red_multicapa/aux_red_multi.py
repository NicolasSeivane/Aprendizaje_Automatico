import numpy as np
import re

def cargar_modelo_completo(nombre_archivo='pesos_cbow_pc2_epoca0.npz'):
    
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


with open("C:\\Users\\PIA\\Documents\\Aprendizaje_Automatico\\Evaluacion_Modelos\\corpus_junto2.txt", "r", encoding="utf-8") as f:
    words = f.read().splitlines()

corpus_modificado = words.copy()

palabras_a_indice = {}
indices_a_palabras = {}
diccionario_onehot = {}
diccionario_onehot_a_palabra = {}
diccionario_conteo = {}
indices_a_embeddings = {}

W1, W2,N, C, eta = cargar_modelo_completo("C:\\Users\\PIA\\Documents\\Aprendizaje_Automatico\\buenpeso\\pesos_cbow_mejores_epoca400_neurona_oculta130.npz")
if W1 is None:
    print('aca esta el problema')

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

def tokenizar_por_vocab(texto, vocab, indices = False):
    palabras = texto.lower()
    palabras = re.findall(r'\w+|[^\w\s]', palabras, flags=re.UNICODE) # tokenización básica por espacios
    tokens = []
    i = 0
    n = len(palabras)

    while i < n:
        cand_final = None
        for j in range(n, i, -1):
            cand = " ".join(palabras[i:j])
            if cand in vocab:
                cand_final = cand
                i = j  
                break
        
        if not cand_final:
            cand_final = palabras[i]
            if cand_final not in vocab:
               return print(f'palabra: [{cand_final}] no esta en voabulario') 
               
            i += 1

        if indices is False:
            tokens.append(cand_final)
        else:
            tokens.append(palabras_a_indice[cand_final])
    return tokens