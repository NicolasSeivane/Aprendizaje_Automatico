# Proyecto de Procesamiento de Lenguaje Natural: Word Embeddings y Generación Literaria

Este repositorio contiene una implementación avanzada de **Procesamiento de Lenguaje Natural (NLP)**, desarrollada en la **Universidad Nacional de Hurlingham (UNAHUR)**. 

El proyecto demuestra el flujo completo desde la creación de representaciones semánticas hasta la generación automática de texto con estilo literario.

---

## 🚀 Estructura del Proyecto (3 Fases)

El repositorio está organizado siguiendo el flujo lógico del desarrollo:

### 1️⃣ Fase 1: Word Embeddings (`01_Word_Embeddings/`)
Implementación de los modelos **CBOW** y **SkipGram** con *Negative Sampling*. 
*   **Propósito**: Crear vectores densos que capturen el significado de las palabras basándose en su contexto.
*   **Contenido**: Scripts de entrenamiento y utilitarios para la lógica de Word2Vec.

### 2️⃣ Fase 2: Entrenamiento de Redes Multicapa - TP2 (`02_Generacion_Texto/`)
Uso de los embeddings obtenidos en la Fase 1 para entrenar modelos predictivos complejos. Esta carpeta representa el núcleo del **Trabajo Práctico 2**.
*   **Propósito**: Entrenar una red neuronal multicapa capaz de modelar el lenguaje de **Julio Cortázar**.
*   **Contenido**: Notebooks de entrenamiento, scripts de arquitectura y una consola de pruebas integrada.

### 3️⃣ Fase 3: Julio Cortázar GPT - Aplicación Final (`03_Consola_Cortazar_GPT/`)
El producto final del proyecto: una interfaz gráfica interactiva independiente.
*   **Propósito**: Generar texto en tiempo real utilizando el mejor modelo consolidado.
*   **Contenido**: Consola interactiva (`consola.py`) y motor de predicción optimizado.

---

## 📂 Organización de Recursos Adicionales

*   📂 **`models/`**: Pesos y parámetros de los modelos entrenados en todas las fases.
*   📂 **`data/`**: Corpus lingüísticos (textos de Cortázar) y datasets.
*   📂 **`trabajos_practicos/`**: Carpeta centralizada con las entregas académicas (TP1, TP2, TP3).
*   📂 **`reports/`**: Informes técnicos en PDF con el análisis detallado de cada desarrollo.

---

## 🧠 Modelos y Tecnologías
*   **Arquitecturas**: Word2Vec (CBOW/SkipGram), Redes Neuronales Multicapa (MLP).
*   **Librerías**: TensorFlow/Keras, NumPy, Scikit-learn, Matplotlib, Pillow (GUI).
*   **Generación Creativa**: Implementación de **Top-k Sampling** para introducir variabilidad literaria y evitar bucles infinitos.

---

## 🛠️ Instalación y Uso

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/Aprendizaje_Automatico.git
    cd Aprendizaje_Automatico
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install numpy tensorflow scikit-learn matplotlib pillow
    ```

3.  **Ejecutar la Consola de Generación:**
    ```bash
    cd 03_Consola_Cortazar_GPT
    python consola.py
    ```

---

## 👥 Autores
*   **Seivane Nicolás**
*   **Cisnero Matías**
*   **Serafini Franco**

---
> [!NOTE]
> Este proyecto fue desarrollado bajo la supervisión académica de la **Universidad Nacional de Hurlingham**.