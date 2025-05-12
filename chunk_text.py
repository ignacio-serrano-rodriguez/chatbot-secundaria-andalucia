import os
import glob
import json
import sys

def create_chunks(text, chunk_size, chunk_overlap):
    """
    Splits the text into overlapping chunks.
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        start += chunk_size - chunk_overlap
        # Ensure we don't go past the end if overlap is large or chunk_size is small
        if start >= text_len: 
            break
            
    # If the last chunk was created by stepping back due to overlap,
    # and it doesn't reach the end of the text, we might miss a small tail.
    # This check ensures the very end of the text is included if not already covered.
    # However, the current logic should cover it as `end` can go beyond `text_len`
    # and Python slicing handles that gracefully.
    # A simpler way is to ensure the last chunk always goes to text_len if it's the final one.
    # The current loop structure should handle this correctly.
    # Let's refine the loop slightly for clarity on the last chunk.

    # Re-evaluating the loop for robustness:
    chunks = []
    current_position = 0
    while current_position < len(text):
        end_position = current_position + chunk_size
        chunks.append(text[current_position:end_position])
        current_position += chunk_size - chunk_overlap
        if current_position >= len(text) and len(text) > (current_position - (chunk_size - chunk_overlap) + chunk_size) : # Avoid re-adding if last chunk perfectly ended
             # This condition is tricky, let's simplify.
             # If the step forward (current_position) is beyond the start of the last chunk, break.
             pass # The loop condition `current_position < len(text)` handles termination.

    # A more standard way to implement fixed-size chunking with overlap:
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(text): # Ensure the last chunk captures the end
            break 
    # If the loop finishes and the last chunk didn't reach the end due to step size
    if chunks and len(text) > len("".join(chunks).replace("","")) - (len(chunks)-1)*chunk_overlap : # A bit complex check
        # A simpler check: if the end of the last chunk is before the end of the text
        last_chunk_end_char_index = (len(chunks) -1) * (chunk_size - chunk_overlap) + chunk_size
        if last_chunk_end_char_index < len(text) and len(text) > chunk_size : # if text is smaller than chunk_size, one chunk is enough
             # This can happen if the last step (chunk_size - chunk_overlap) jumps over the end.
             # Let's ensure the final part is always captured if it's substantial.
             # The range step handles this well, but the last chunk might be smaller.
             # The current for loop is generally good. The last chunk will be text[i:i+chunk_size]
             # which might be shorter than chunk_size if i+chunk_size > len(text). This is fine.
             pass


    return chunks

def main():
    """
    Main function to find cleaned TXT files, chunk their content,
    and save chunks to JSON files.
    """
    input_directory = "TXTs_cleaned"
    output_directory = "Chunks"
    
    # Define chunking parameters
    chunk_size = 1000  # characters
    chunk_overlap = 200 # characters

    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(output_directory):
        try:
            os.makedirs(output_directory)
            print(f"Created output directory: '{output_directory}'")
        except OSError as e:
            print(f"Error creating output directory '{output_directory}': {e}", file=sys.stderr)
            sys.exit(1)

    txt_files = glob.glob(os.path.join(input_directory, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{input_directory}'.")
        return

    print(f"Found {len(txt_files)} .txt files in '{input_directory}'. Processing...")

    for txt_path in txt_files:
        print(f"\nProcessing: {txt_path}")
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                cleaned_text = file.read()
        except Exception as e:
            print(f"Error reading file '{txt_path}': {e}", file=sys.stderr)
            continue

        text_chunks = create_chunks(cleaned_text, chunk_size, chunk_overlap)
        
        if not text_chunks:
            print(f"No chunks generated for {txt_path}. Text might be too short or empty.")
            continue

        base_filename = os.path.basename(txt_path)
        json_filename = os.path.splitext(base_filename)[0] + "_chunks.json"
        json_path = os.path.join(output_directory, json_filename)

        try:
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(text_chunks, json_file, ensure_ascii=False, indent=4)
            print(f"Successfully created {len(text_chunks)} chunks and saved to: {json_path}")
        except Exception as e:
            print(f"Error writing JSON file '{json_path}': {e}", file=sys.stderr)

    print("\nText chunking complete.")

if __name__ == "__main__":
    # Before running, ensure you have run preprocess_text.py
    # and the TXTs_cleaned directory contains the cleaned text files.
    main()
