import os
import glob
import json
import sys
from sentence_transformers import SentenceTransformer # type: ignore

def generate_embeddings_for_chunks(chunks, model):
    """
    Generates embeddings for a list of text chunks using the provided model.

    Args:
        chunks (list): A list of strings, where each string is a text chunk.
        model (SentenceTransformer): The pre-loaded sentence transformer model.

    Returns:
        list: A list of embeddings (list of floats) corresponding to the input chunks.
               Returns an empty list if an error occurs during embedding.
    """
    try:
        embeddings = model.encode(chunks, show_progress_bar=True)
        return embeddings.tolist() # Convert numpy arrays to lists for JSON serialization
    except Exception as e:
        print(f"Error during embedding generation: {e}", file=sys.stderr)
        return []

def main():
    """
    Main function to find chunked JSON files, generate embeddings,
    and save them to a new directory.
    Prerequisites for running this script:
    1. You have run chunk_text.py and the 'Chunks/' directory is populated with *_chunks.json files.
    2. The sentence-transformers library is installed (e.g., pip install sentence-transformers).
       The script will attempt to download the specified model if it's not cached locally.
    """
    input_directory = "Chunks"
    output_directory = "Embeddings"
    model_name = 'all-MiniLM-L6-v2' # As specified in plan.md

    # Ensure the input directory exists
    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' not found.", file=sys.stderr)
        print("Please ensure you have run 'chunk_text.py' first.", file=sys.stderr)
        sys.exit(1)

    # Ensure the output directory exists, create it if not
    if not os.path.isdir(output_directory):
        try:
            os.makedirs(output_directory)
            print(f"Created output directory: '{output_directory}'")
        except OSError as e:
            print(f"Error creating output directory '{output_directory}': {e}", file=sys.stderr)
            sys.exit(1)

    # Load the sentence transformer model
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading sentence transformer model '{model_name}': {e}", file=sys.stderr)
        print("Make sure you have internet access to download the model for the first time, or that the model is cached.", file=sys.stderr)
        print("You might need to install the library: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)

    # Find all JSON files in the input directory (from chunk_text.py)
    chunk_files = glob.glob(os.path.join(input_directory, "*_chunks.json"))

    if not chunk_files:
        print(f"No '*_chunks.json' files found in '{input_directory}'.")
        return

    print(f"Found {len(chunk_files)} chunk files in '{input_directory}'. Processing...")

    for chunk_file_path in chunk_files:
        print(f"\nProcessing file: {chunk_file_path}")
        try:
            with open(chunk_file_path, 'r', encoding='utf-8') as f:
                list_of_chunk_texts = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file '{chunk_file_path}': {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error reading file '{chunk_file_path}': {e}", file=sys.stderr)
            continue

        if not isinstance(list_of_chunk_texts, list) or not all(isinstance(chunk, str) for chunk in list_of_chunk_texts):
            print(f"Warning: File '{chunk_file_path}' does not contain a list of strings. Skipping.", file=sys.stderr)
            continue
        
        if not list_of_chunk_texts:
            print(f"No text chunks found in '{chunk_file_path}'. Skipping.")
            continue

        print(f"Generating embeddings for {len(list_of_chunk_texts)} chunks from {os.path.basename(chunk_file_path)}...")
        chunk_embeddings = generate_embeddings_for_chunks(list_of_chunk_texts, model)

        if not chunk_embeddings or len(chunk_embeddings) != len(list_of_chunk_texts):
            print(f"Failed to generate embeddings for all chunks in '{chunk_file_path}'. Skipping output for this file.", file=sys.stderr)
            continue

        output_data = []
        for text_chunk, embedding_vector in zip(list_of_chunk_texts, chunk_embeddings):
            output_data.append({
                "chunk_text": text_chunk,
                "embedding": embedding_vector
            })

        base_filename = os.path.basename(chunk_file_path)
        output_filename = base_filename.replace("_chunks.json", "_embeddings.json")
        output_file_path = os.path.join(output_directory, output_filename)

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                json.dump(output_data, f_out, ensure_ascii=False, indent=4)
            print(f"Successfully generated embeddings and saved to: {output_file_path}")
        except Exception as e:
            print(f"Error writing embeddings JSON file '{output_file_path}': {e}", file=sys.stderr)

    print("\nEmbedding generation complete.")

if __name__ == "__main__":
    main()
