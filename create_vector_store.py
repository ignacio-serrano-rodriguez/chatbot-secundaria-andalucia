import os
import glob
import json
import numpy as np
import faiss # type: ignore
import sys

def get_embedding_dimension(model_name='all-MiniLM-L6-v2'):
    """
    Helper function to get the embedding dimension for a sentence-transformer model.
    This is a bit of a workaround as we don't want to load the model if not necessary,
    but for common models, dimensions are known.
    For 'all-MiniLM-L6-v2', it's 384.
    """
    if model_name == 'all-MiniLM-L6-v2':
        return 384
    else:
        # Fallback or raise error if model is different and dimension unknown
        print(f"Warning: Unknown model '{model_name}'. Assuming dimension 384. For other models, this might be incorrect.", file=sys.stderr)
        # For a more robust solution, you might load the model here to get its dimension,
        # but that adds overhead if this script is run frequently.
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(model_name)
        # return model.get_sentence_embedding_dimension()
        return 384

def main():
    """
    Main function to load embeddings and create a FAISS vector store.
    """
    embeddings_input_dir = "Embeddings"
    vector_store_output_dir = "vector_store"
    faiss_index_filename = "faiss_index.index"
    metadata_filename = "doc_chunks_metadata.json"

    embedding_dim = get_embedding_dimension() # For 'all-MiniLM-L6-v2'

    # Ensure the embeddings input directory exists
    if not os.path.isdir(embeddings_input_dir):
        print(f"Error: Embeddings directory '{embeddings_input_dir}' not found.", file=sys.stderr)
        print("Please ensure you have run 'generate_embeddings.py' first.", file=sys.stderr)
        sys.exit(1)

    # Ensure the vector store output directory exists, create it if not
    if not os.path.isdir(vector_store_output_dir):
        try:
            os.makedirs(vector_store_output_dir)
            print(f"Created output directory: '{vector_store_output_dir}'")
        except OSError as e:
            print(f"Error creating output directory '{vector_store_output_dir}': {e}", file=sys.stderr)
            sys.exit(1)

    embedding_files = glob.glob(os.path.join(embeddings_input_dir, "*_embeddings.json"))

    if not embedding_files:
        print(f"No '*_embeddings.json' files found in '{embeddings_input_dir}'. Nothing to index.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(embedding_files)} embedding files. Processing...")

    all_embeddings = []
    all_chunk_texts = [] # To store the original text chunks

    for emb_file_path in embedding_files:
        print(f"Reading embeddings from: {emb_file_path}")
        try:
            with open(emb_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f) # This is a list of {"chunk_text": "...", "embedding": [...]}
            
            if not isinstance(data, list):
                print(f"Warning: Expected a list in {emb_file_path}, got {type(data)}. Skipping.", file=sys.stderr)
                continue

            for item in data:
                if isinstance(item, dict) and "chunk_text" in item and "embedding" in item:
                    all_chunk_texts.append(item["chunk_text"])
                    all_embeddings.append(item["embedding"])
                else:
                    print(f"Warning: Skipping invalid item in {emb_file_path}: {item}", file=sys.stderr)
        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file '{emb_file_path}': {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error reading or processing file '{emb_file_path}': {e}", file=sys.stderr)
            continue
            
    if not all_embeddings:
        print("No valid embeddings collected from files. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Convert embeddings to a NumPy array
    try:
        embeddings_np = np.array(all_embeddings, dtype='float32')
        if embeddings_np.ndim != 2 or embeddings_np.shape[1] != embedding_dim:
            print(f"Error: Embeddings shape is {embeddings_np.shape}, expected (n_chunks, {embedding_dim}).", file=sys.stderr)
            print("Please check the embedding generation process and model.", file=sys.stderr)
            sys.exit(1)
    except ValueError as e:
        print(f"Error converting embeddings to NumPy array: {e}", file=sys.stderr)
        print("This might be due to inconsistent embedding dimensions or non-numeric data.", file=sys.stderr)
        sys.exit(1)


    print(f"Collected {len(all_chunk_texts)} chunks and {embeddings_np.shape[0]} embeddings.")
    print(f"Embeddings array shape: {embeddings_np.shape}")

    # Create FAISS index
    # Using IndexFlatL2 for simple L2 distance search.
    # For larger datasets, more complex indexes like IndexIVFFlat might be better.
    index = faiss.IndexFlatL2(embedding_dim)
    
    print(f"FAISS index created. Is_trained: {index.is_trained}, Total vectors: {index.ntotal}")

    # Add embeddings to the index
    try:
        index.add(embeddings_np)
        print(f"Embeddings added to FAISS index. Total vectors in index: {index.ntotal}")
    except Exception as e:
        print(f"Error adding embeddings to FAISS index: {e}", file=sys.stderr)
        sys.exit(1)

    # Save the FAISS index
    faiss_index_path = os.path.join(vector_store_output_dir, faiss_index_filename)
    try:
        faiss.write_index(index, faiss_index_path)
        print(f"FAISS index saved to: {faiss_index_path}")
    except Exception as e:
        print(f"Error saving FAISS index to '{faiss_index_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Save the corresponding chunk texts (metadata)
    metadata_path = os.path.join(vector_store_output_dir, metadata_filename)
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f_meta:
            json.dump(all_chunk_texts, f_meta, ensure_ascii=False, indent=4)
        print(f"Document chunks metadata saved to: {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata to '{metadata_path}': {e}", file=sys.stderr)
        sys.exit(1)
        
    print("\nVector store creation complete.")
    print(f"Index file: {faiss_index_path}")
    print(f"Metadata file: {metadata_path}")

if __name__ == "__main__":
    # Before running, ensure:
    # 1. You have run generate_embeddings.py and the 'Embeddings/' directory is populated.
    # 2. FAISS and NumPy are installed: pip install faiss-cpu numpy
    #    (faiss-gpu can be used if you have a compatible GPU and CUDA setup)
    main()
