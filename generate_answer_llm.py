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
    Generates an answer from a local LLM using the Ollama API.

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

    # Construct the prompt
    context_str = "\n\n".join(context_chunks)
    prompt = f"""
    Contexto:
    {context_str}

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
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API at {ollama_url}: {e}")
        print("Please ensure Ollama is running and accessible.")
        print("You can start Ollama by running 'ollama serve' in your terminal.")
        print("If you are using a specific model like 'mistral:7b-instruct', ensure it is pulled with 'ollama pull mistral:7b-instruct'.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from Ollama API: {e}")
        print(f"Raw response content: {response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    print("--- LLM Answer Generation Script (Example Usage) ---")

    # This is an example. In a real application, the query and context
    # would come from the user and the retrieval system respectively.

    # 1. Example User Query
    example_query = "Explícame los criterios de evaluación en la Educación Secundaria Obligatoria según la normativa."

    # 2. Example Context Chunks (simulating what query_and_retrieve.py would provide)
    # In a real scenario, these would be dynamically fetched based on the query.
    example_context = [
        "La evaluación del proceso de aprendizaje del alumnado de Educación Secundaria Obligatoria será continua, formativa e integradora.",
        "Se establecerán medidas de refuerzo educativo para el alumnado que presente dificultades de aprendizaje, con especial atención a la adquisición de las competencias clave.",
        "Los referentes para la evaluación de las competencias específicas de las materias serán los criterios de evaluación.",
        "El profesorado evaluará tanto los aprendizajes del alumnado como los procesos de enseñanza y su propia práctica docente.",
        "Los resultados de la evaluación se expresarán en los términos Insuficiente (IN), Suficiente (SU), Bien (BI), Notable (NT), o Sobresaliente (SB)."
    ]

    print(f"\nUser Query: {example_query}")
    print(f"Number of context chunks: {len(example_context)}")

    # 3. Generate the answer
    # You might need to adjust OLLAMA_MODEL_NAME if "mistral:7b-instruct" is not available
    # or if you prefer another model you have pulled with Ollama.
    generated_answer = generate_llm_answer(example_query, example_context)

    if generated_answer:
        print("\n--- Generated LLM Answer ---")
        print(generated_answer)
    else:
        print("\n--- Failed to generate LLM Answer ---")
        print("Please check the console output for error messages.")
        print("Ensure Ollama is running and the specified model is available (e.g., 'ollama pull mistral:7b-instruct').")

    print("\n--- Script Finished ---")
