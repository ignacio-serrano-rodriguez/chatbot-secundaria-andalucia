import os
import json
import numpy as np
import faiss # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import sys

# Configuration
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_FILENAME = "faiss_index.index"
METADATA_FILENAME = "doc_chunks_metadata.json"
MODEL_NAME = 'all-MiniLM-L6-v2'

def load_retrieval_components(model_name, vector_store_dir, faiss_index_filename, metadata_filename):
    """
    Loads the sentence transformer model, FAISS index, and metadata.

    Returns:
        tuple: (model, index, metadata_chunks) or (None, None, None) if an error occurs.
    """
    print(f"Loading sentence transformer model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading sentence transformer model '{model_name}': {e}", file=sys.stderr)
        print("Make sure you have internet access or the model is cached.", file=sys.stderr)
        print("You might need to install the library: pip install sentence-transformers", file=sys.stderr)
        return None, None, None

    faiss_index_path = os.path.join(vector_store_dir, faiss_index_filename)
    metadata_path = os.path.join(vector_store_dir, metadata_filename)

    if not os.path.exists(faiss_index_path):
        print(f"Error: FAISS index file not found at '{faiss_index_path}'.", file=sys.stderr)
        print("Please ensure 'create_vector_store.py' has been run successfully.", file=sys.stderr)
        return model, None, None

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at '{metadata_path}'.", file=sys.stderr)
        print("Please ensure 'create_vector_store.py' has been run successfully.", file=sys.stderr)
        return model, None, None

    print(f"Loading FAISS index from: {faiss_index_path}...")
    try:
        index = faiss.read_index(faiss_index_path)
        print(f"FAISS index loaded. Total vectors: {index.ntotal}")
    except Exception as e:
        print(f"Error loading FAISS index: {e}", file=sys.stderr)
        return model, None, None

    print(f"Loading metadata from: {metadata_path}...")
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_chunks = json.load(f)
        print(f"Metadata loaded. Number of chunks: {len(metadata_chunks)}")
    except Exception as e:
        print(f"Error loading metadata: {e}", file=sys.stderr)
        return model, index, None
        
    if index.ntotal != len(metadata_chunks):
        print(f"Warning: FAISS index ({index.ntotal} vectors) and metadata ({len(metadata_chunks)} chunks) mismatch!", file=sys.stderr)
        # Decide how to handle this: exit, or proceed with caution
        # For now, we'll proceed but this indicates a potential issue in the vector store creation.

    return model, index, metadata_chunks

def retrieve_relevant_chunks(query, model, index, metadata_chunks, k=5):
    """
    Embeds the query, searches the FAISS index, and retrieves relevant chunks.

    Args:
        query (str): The user's query.
        model (SentenceTransformer): The loaded sentence transformer model.
        index (faiss.Index): The loaded FAISS index.
        metadata_chunks (list): The list of original text chunks.
        k (int): The number of top relevant chunks to retrieve.

    Returns:
        list: A list of the top k relevant text chunks, or an empty list if an error occurs.
    """
    if not query or not model or not index or not metadata_chunks:
        print("Error: Missing one or more required components for retrieval.", file=sys.stderr)
        return []

    print(f"\nEmbedding query: \"{query}\"")
    try:
        query_embedding = model.encode([query], convert_to_numpy=True)
    except Exception as e:
        print(f"Error embedding query: {e}", file=sys.stderr)
        return []

    print(f"Searching FAISS index for top {k} similar chunks...")
    try:
        # D: distances, I: indices
        distances, indices = index.search(query_embedding, k)
    except Exception as e:
        print(f"Error searching FAISS index: {e}", file=sys.stderr)
        return []

    retrieved_chunks = []
    print("\nRetrieved chunks:")
    if indices.size == 0 or indices[0][0] == -1 : # -1 can indicate no results or error in some FAISS versions/setups
        print("No relevant chunks found.")
        return []

    for i in range(len(indices[0])):
        idx = indices[0][i]
        if 0 <= idx < len(metadata_chunks):
            chunk_text = metadata_chunks[idx]
            retrieved_chunks.append(chunk_text)
            # print(f"  --- Chunk {i+1} (Index: {idx}, Distance: {distances[0][i]:.4f}) ---")
            # print(chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text) # Print a snippet
            # print("-" * 30)
        else:
            print(f"Warning: Retrieved index {idx} is out of bounds for metadata (size {len(metadata_chunks)}).", file=sys.stderr)
            
    return retrieved_chunks

def main():
    """
    Main function to demonstrate query processing and retrieval.
    """
    print("--- Query Processing and Retrieval Script ---")
    
    model, index, metadata_chunks = load_retrieval_components(
        MODEL_NAME,
        VECTOR_STORE_DIR,
        FAISS_INDEX_FILENAME,
        METADATA_FILENAME
    )

    if not model or not index or not metadata_chunks:
        print("\nExiting due to loading errors.", file=sys.stderr)
        sys.exit(1)

    # Example query
    # You can change this or take it as input from the user
    sample_query = "Explícame la evaluación en Educación Secundaria Obligatoria"
    
    print(f"\nAttempting to retrieve chunks for query: \"{sample_query}\"")
    
    relevant_chunks = retrieve_relevant_chunks(sample_query, model, index, metadata_chunks, k=3)

    if relevant_chunks:
        print(f"\n--- Top {len(relevant_chunks)} Relevant Chunks for '{sample_query}' ---")
        for i, chunk in enumerate(relevant_chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk)
            print("-" * 50)
    else:
        print(f"\nNo relevant chunks were retrieved for the query: '{sample_query}'")

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    # Before running, ensure:
    # 1. You have run 'create_vector_store.py' and the 'vector_store/' directory is populated.
    # 2. Required libraries are installed:
    #    pip install sentence-transformers faiss-cpu numpy
    #    (Use faiss-gpu if you have a compatible GPU and CUDA setup)
    main()
