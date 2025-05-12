# Chatbot de las leyes educativas de secundaria en andalucía

Este documento describe la creación de un chatbot local que procesa varios archivos PDF para responder a las preguntas de los usuarios. El proyecto se ha desarrollado construyendo manualmente cada componente de la canalización RAG (Retrieval Augmented Generation). Este enfoque ofrece el máximo control y un profundo conocimiento de cada paso. El desarrollo ha sido iterativo, con un perfeccionamiento continuo.

1.  **Extracción de texto PDF**

    - Utiliza la librería `PyMuPDF` (Fitz) en Python (`extract_pdf_text.py`). Los PDFs se procesan desde el directorio `PDFs/`, y el texto extraído se guarda en el directorio `TXTs/`.

2.  **Preprocesamiento y limpieza de texto:**

    - Un script de Python (`preprocess_text.py`) realiza la limpieza básica (por ejemplo, normalizar los espacios en blanco). El texto limpiado de `TXTs/` se guarda en `TXTs_cleaned/`.

3.  **Agrupación de textos:**

    - Un script en Python (`chunk_text.py`) divide el texto en trozos de caracteres de tamaño fijo con solapamiento. Los trozos de `TXTs_cleaned/` se guardan como archivos JSON en el directorio `Chunks/`.

4.  **Generación de la incrustación:**

    - Utilización de la librería `sentence-transformers` con el modelo `all-MiniLM-L6-v2` (`generate_embeddings.py`). Las incrustaciones para los trozos de `Chunks/` se generan y guardan como archivos JSON en el directorio `Embeddings/`. Este paso se ejecuta localmente.

5.  **Creación de almacenes vectoriales**

    - Utilización de `FAISS` (Facebook AI Similarity Search) para crear un índice (`create_vector_store.py`). Las incrustaciones de `Embeddings/` y los metadatos de los trozos correspondientes (de `Chunks/`) se utilizan para construir el índice, que se guarda en el directorio `vector_store/` (`faiss_index.index` y `doc_chunks_metadata.json`).

6.  **Procesamiento y recuperación de consultas:**

    - La consulta del usuario se incrusta utilizando el mismo modelo `all-MiniLM-L6-v2`. Se busca en el índice FAISS los trozos más similares (`query_and_retrieve.py`).

7.  **Generación de respuestas con LLM:**

    - **LLM local:** Utiliza un Large Language Model local, concretamente `mistral:7b-instruct`, al que se accede a través de `Ollama` (`generate_answer_llm.py`).

    - **Prompt engineering:** Una instrucción específica ordena al LLM que responda _únicamente_ basándose en el contexto proporcionado y que indique si la información no se encuentra.

    - **Gestión del contexto:** Los trozos recuperados se pasan como contexto al LLM.

8.  **Interfaz de usuario**

    - Una interfaz de línea de comandos (CLI) es proporcionada por el script principal `chatbot.py`. Se le pide al usuario con "Yo: " y muestra las respuestas del bot con el prefijo "Chatbot: ".

9.  **Orquestación:**

    - El script `chatbot.py` orquesta todos los pasos, desde la preparación de datos (con una opción para forzar el reprocesamiento) hasta el bucle de chat interactivo. Llama directamente a funciones de los otros módulos.

**Resumen del flujo**

PDFs en `PDFs/` -> `extract_pdf_text.py` (PyMuPDF) -> Texto en `TXTs/` -> `preprocess_text.py` -> Texto limpio en `TXTs_cleaned/` -> `chunk_text.py` -> Chunks en `Chunks/` -> `generate_embeddings.py` (Sentence Transformers `all-MiniLM-L6-v2`) -> Embeddings en `Embeddings/` -> `create_vector_store.py` (FAISS) -> Índices en `vector_store/` -> Pregunta del usuario (CLI en `chatbot.py`) -> Pregunta embebida -> Búsqueda en FAISS Index -> Devolución de chunks relevantes -> Alimentación del contexto + pregunta a `generate_answer_llm.py` (Ollama with `mistral:7b-instruct`) -> Mostrar respuesta en CLI.
