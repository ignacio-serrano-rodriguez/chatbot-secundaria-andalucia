'''
This script defines a function to generate answers from a local LLM 
using the Ollama API based on a user query and provided context chunks.
'''
import json
import requests

# Configuration for the Ollama API
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Replace with your desired Ollama model, e.g., "mistral:7b-instruct", "llama2", "gemma:2b"
OLLAMA_MODEL_NAME = "mistral:7b-instruct" 

def generate_llm_answer(query: str, context_chunks: list[str], model_name: str = OLLAMA_MODEL_NAME, ollama_url: str = OLLAMA_API_URL) -> str | None:
    """
    Generates an answer from a local LLM using the Ollama API,
    strictly based on the provided context.
    If the answer is not in the context, it should indicate so.

    Args:
        query (str): The user's query.
        context_chunks (list[str]): A list of text chunks to provide as context.
        model_name (str): The name of the Ollama model to use.
        ollama_url (str): The URL of the Ollama API.

    Returns:
        str | None: The LLM's generated answer as a string, or None if an error occurs.
    """
    if not query:
        print("Error: Query cannot be empty.")
        return None
    if not context_chunks:
        print("Warning: Context chunks list is empty. Proceeding with query only.")

    # Construct the prompt with specific instructions
    context_str = "\n\n".join(context_chunks)
    prompt = f"""
    Por favor, responde la siguiente pregunta basándote ÚNICAMENTE en el contexto proporcionado a continuación.
    No utilices ningún conocimiento externo o general que puedas tener.
    Si la respuesta a la pregunta no se encuentra explícitamente en el contexto, debes indicar claramente que no puedes responder con la información disponible en los documentos o que la información no se encuentra en el contexto.

    Contexto:
    ---
    {context_str if context_chunks else "No se ha proporcionado ningún contexto de los documentos."}
    ---

    Pregunta: {query}

    Respuesta:
    """

    # Prepare the data payload for the Ollama API
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # Get the full response at once
    }

    print(f"\nSending request to Ollama API with model: {model_name}...")
    # print(f"Prompt being sent (first 200 chars): {prompt[:200]}...") # For debugging

    try:
        response = requests.post(ollama_url, json=payload, timeout=120) # 120 seconds timeout
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        # Process the JSON response
        # Ollama returns a sequence of JSON objects if stream=True, 
        # or a single JSON object if stream=False.
        response_data = response.json()
        
        if "response" in response_data:
            full_answer = response_data["response"].strip()
            print("Successfully received answer from LLM.")
            return full_answer
        else:
            print(f"Error: 'response' key not found in Ollama API response. Full response: {response_data}")
            return "Lo siento, no pude procesar la respuesta del modelo correctamente."

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API at {ollama_url}: {e}")
        print("Please ensure Ollama is running and accessible.")
        print("You can start Ollama by running 'ollama serve' in your terminal.")
        print(f"If you are using a specific model like '{model_name}', ensure it is pulled with 'ollama pull {model_name}'.")
        return "Lo siento, no pude conectarme con el modelo de lenguaje. Por favor, verifica que Ollama esté funcionando."
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from Ollama API: {e}")
        print(f"Raw response content: {response.text}")
        return "Lo siento, hubo un problema al decodificar la respuesta del modelo."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "Lo siento, ocurrió un error inesperado al generar la respuesta."

if __name__ == "__main__":
    print("--- LLM Answer Generation Script (Example Usage with Strict Context) ---")

    example_query_in_context = "Explícame los criterios de evaluación en la Educación Secundaria Obligatoria según la normativa."
    example_context_for_query = [
        "La evaluación del proceso de aprendizaje del alumnado de Educación Secundaria Obligatoria será continua, formativa e integradora.",
        "Se establecerán medidas de refuerzo educativo para el alumnado que presente dificultades de aprendizaje, con especial atención a la adquisición de las competencias clave.",
        "Los referentes para la evaluación de las competencias específicas de las materias serán los criterios de evaluación.",
        "El profesorado evaluará tanto los aprendizajes del alumnado como los procesos de enseñanza y su propia práctica docente.",
        "Los resultados de la evaluación se expresarán en los términos Insuficiente (IN), Suficiente (SU), Bien (BI), Notable (NT), o Sobresaliente (SB)."
    ]

    example_query_not_in_context = "¿Cuál es la capital de Francia?" # This should not be answerable from the context

    print(f"\nUser Query (in context): {example_query_in_context}")
    generated_answer_1 = generate_llm_answer(example_query_in_context, example_context_for_query)
    if generated_answer_1:
        print("\n--- Generated LLM Answer 1 ---")
        print(generated_answer_1)
    else:
        print("\n--- Failed to generate LLM Answer 1 ---")

    print(f"\nUser Query (NOT in context): {example_query_not_in_context}")
    generated_answer_2 = generate_llm_answer(example_query_not_in_context, example_context_for_query)
    if generated_answer_2:
        print("\n--- Generated LLM Answer 2 ---")
        print(generated_answer_2) # Expected: "I don't know" or similar
    else:
        print("\n--- Failed to generate LLM Answer 2 ---")

    print(f"\nUser Query (with NO context provided to LLM): {example_query_in_context}")
    generated_answer_3 = generate_llm_answer(example_query_in_context, []) # Simulating no relevant chunks found
    if generated_answer_3:
        print("\n--- Generated LLM Answer 3 ---")
        print(generated_answer_3) # Expected: "I don't know" or similar
    else:
        print("\n--- Failed to generate LLM Answer 3 ---")

    print("\n--- Script Finished ---")
