"""
Main chatbot application script.
Orchestrates the PDF processing pipeline and handles user interaction.
"""
import os
import sys
import subprocess
import importlib.util # To import functions from other scripts

# Import functions from your existing scripts
# We'll use this method to directly call functions instead of subprocesses for better integration
from extract_pdf_text import main as extract_pdf_text_main
from preprocess_text import main as preprocess_text_main
from chunk_text import main as chunk_text_main
from generate_embeddings import main as generate_embeddings_main
from create_vector_store import main as create_vector_store_main
from query_and_retrieve import load_retrieval_components, retrieve_relevant_chunks
from generate_answer_llm import generate_llm_answer

# Configuration (can be moved to a config file later)
PDF_DIR = "PDFs"
TXT_DIR = "TXTs"
TXT_CLEANED_DIR = "TXTs_cleaned"
CHUNKS_DIR = "Chunks"
EMBEDDINGS_DIR = "Embeddings"
VECTOR_STORE_DIR = "vector_store"

# For query_and_retrieve and generate_answer_llm
MODEL_NAME = 'all-MiniLM-L6-v2' # Embedding model
OLLAMA_MODEL_NAME = "mistral:7b-instruct" # LLM model

# --- Helper Functions ---

def check_and_create_dir(directory_path):
    """Checks if a directory exists, and creates it if it doesn't."""
    if not os.path.isdir(directory_path):
        print(f"Directory '{directory_path}' not found. Creating it...")
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as e:
            print(f"Error creating directory '{directory_path}': {e}", file=sys.stderr)
            sys.exit(1) # Exit if we can't create essential directories

def is_pipeline_stage_complete(output_dir_or_file, stage_name):
    """
    Checks if a pipeline stage is complete by verifying if its output directory/file exists
    and is not empty (for directories).
    """
    if not os.path.exists(output_dir_or_file):
        print(f"Stage '{stage_name}' output ('{output_dir_or_file}') not found.")
        return False
    if os.path.isdir(output_dir_or_file) and not os.listdir(output_dir_or_file):
        print(f"Stage '{stage_name}' output directory ('{output_dir_or_file}') is empty.")
        return False
    return True

# --- Main Processing Functions ---

def run_data_preparation_pipeline(force_rerun=False):
    """
    Runs the data preparation pipeline:
    1. PDF Text Extraction
    2. Text Preprocessing
    3. Text Chunking
    4. Embedding Generation
    5. Vector Store Creation
    Skips stages if their output already exists, unless force_rerun is True.
    """

    # Create necessary base directories if they don't exist
    check_and_create_dir(PDF_DIR) # User needs to put PDFs here
    check_and_create_dir(TXT_DIR)
    check_and_create_dir(TXT_CLEANED_DIR)
    check_and_create_dir(CHUNKS_DIR)
    check_and_create_dir(EMBEDDINGS_DIR)
    check_and_create_dir(VECTOR_STORE_DIR)

    # Stage 1: PDF Text Extraction
    if force_rerun or not is_pipeline_stage_complete(TXT_DIR, "PDF Text Extraction"):
        print("\nRunning PDF Text Extraction...")
        try:
            extract_pdf_text_main()
        except Exception as e:
            print(f"Error during PDF Text Extraction: {e}", file=sys.stderr)
            return False
    
    # Stage 2: Text Preprocessing
    if force_rerun or not is_pipeline_stage_complete(TXT_CLEANED_DIR, "Text Preprocessing"):
        print("\nRunning Text Preprocessing...")
        try:
            preprocess_text_main()
        except Exception as e:
            print(f"Error during Text Preprocessing: {e}", file=sys.stderr)
            return False

    # Stage 3: Text Chunking
    if force_rerun or not is_pipeline_stage_complete(CHUNKS_DIR, "Text Chunking"):
        print("\nRunning Text Chunking...")
        try:
            chunk_text_main()
        except Exception as e:
            print(f"Error during Text Chunking: {e}", file=sys.stderr)
            return False

    # Stage 4: Embedding Generation
    if force_rerun or not is_pipeline_stage_complete(EMBEDDINGS_DIR, "Embedding Generation"):
        print("\nRunning Embedding Generation...")
        try:
            generate_embeddings_main()
        except Exception as e:
            print(f"Error during Embedding Generation: {e}", file=sys.stderr)
            return False
            
    # Stage 5: Vector Store Creation
    # Check for both index file and metadata file
    faiss_index_file = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
    metadata_file = os.path.join(VECTOR_STORE_DIR, "doc_chunks_metadata.json")
    if force_rerun or not (is_pipeline_stage_complete(faiss_index_file, "Vector Store Index") and \
                           is_pipeline_stage_complete(metadata_file, "Vector Store Metadata")):
        print("\nRunning Vector Store Creation...")
        try:
            create_vector_store_main()
        except Exception as e:
            print(f"Error during Vector Store Creation: {e}", file=sys.stderr)
            return False

    return True

# --- Chatbot Interaction ---
class Chatbot:
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.metadata_chunks = None
        self.retrieval_ready = False

    def load_retrieval_system(self):
        """Loads the components needed for query retrieval."""
        self.embedding_model, self.faiss_index, self.metadata_chunks = load_retrieval_components(
            MODEL_NAME,
            VECTOR_STORE_DIR,
            "faiss_index.index",
            "doc_chunks_metadata.json"
        )
        if self.embedding_model and self.faiss_index and self.metadata_chunks:
            self.retrieval_ready = True
        else:
            print("Failed to load retrieval system. Please check previous errors.", file=sys.stderr)
            self.retrieval_ready = False
        return self.retrieval_ready

    def get_answer(self, query):
        """
        Processes a query: retrieves relevant chunks and generates an answer using LLM.
        """
        if not self.retrieval_ready:
            print("Retrieval system not ready. Cannot process query.", file=sys.stderr)
            return "Error: Retrieval system not initialized."
        
        # 1. Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(
            query, 
            self.embedding_model, 
            self.faiss_index, 
            self.metadata_chunks, 
            k=3 # Number of chunks to retrieve
        )

        if not relevant_chunks:
            return "I couldn't find any relevant information in the documents for your query."
        
        # 2. Generate answer using LLM
        answer = generate_llm_answer(query, relevant_chunks, model_name=OLLAMA_MODEL_NAME)

        if answer:
            return answer
        else:
            return "Sorry, I encountered an error while trying to generate an answer."

def main_chat_loop():
    """
    Main loop for the chatbot interaction.
    """

    # Ask user if they want to force re-run the data preparation pipeline
    force_rerun_input = input("\n¿Desea forzar el reprocesamiento de todos los PDF? (si/no) [no]: ").strip().lower()
    print(f"Cargando chatbot...")
    force_rerun = True if force_rerun_input == 'si' else False
    
    if not run_data_preparation_pipeline(force_rerun=force_rerun):
        print("Chatbot initialization failed due to errors in data preparation.", file=sys.stderr)
        sys.exit(1)

    chatbot = Chatbot()
    if not chatbot.load_retrieval_system():
        print("Chatbot initialization failed: Could not load retrieval system.", file=sys.stderr)
        sys.exit(1)

    print("\nChatbot: Escribe 'adios' o 'salir' para dejar de chatear conmigo.")
    while True:
        try:
            user_query = input("\nYo: ")
            if user_query.lower() in ['adios', 'salir']:
                print("\nChatbot: ¡Nos vemos!")
                break
            if not user_query.strip():
                continue

            bot_answer = chatbot.get_answer(user_query)
            print(f"Chatbot: {bot_answer}")

        except KeyboardInterrupt:
            print("\nChatbot: ¡Nos vemos!")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the chat loop: {e}", file=sys.stderr)
            # Optionally, decide if you want to break the loop or try to continue
            # break 

if __name__ == "__main__":
    # Ensure all script directories are in the Python path
    # This is important if you run chatbot.py from its own directory
    # and the other scripts are in the same directory.
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # sys.path.append(current_dir) # Not strictly necessary if all files are in the same dir and run from there.

    main_chat_loop()
