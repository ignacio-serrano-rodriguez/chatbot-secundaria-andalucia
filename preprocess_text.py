\
import os
import glob
import re
import sys

def clean_text(text):
    """
    Cleans the raw extracted text.
    This is a basic implementation. More sophisticated rules can be added.
    """
    # Normalize whitespace: replace multiple spaces/tabs/newlines with a single space
    text = re.sub(r'\\s+', ' ', text).strip()

    # Example: Remove page numbers if they follow a common pattern
    # This is highly dependent on the PDF structure and might need adjustment
    # text = re.sub(r'Page \\d+ of \\d+', '', text)
    # text = re.sub(r'Página \\d+', '', text) # Spanish example

    # Example: Remove very short lines that might be remnants of headers/footers
    # lines = text.split('\\n')
    # lines = [line for line in lines if len(line.strip()) > 20] # Keep lines longer than 20 chars
    # text = '\\n'.join(lines)

    # Add more cleaning rules here as needed:
    # - Remove specific headers/footers (if they have consistent patterns)
    # - Correct common OCR errors (if applicable)
    # - Handle hyphenated words at the end of lines

    return text

def main():
    """
    Main function to find TXT files, clean their content,
    and save to a new directory.
    """
    input_directory = "TXTs"
    output_directory = "TXTs_cleaned"

    # Ensure the input directory exists
    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' not found.", file=sys.stderr)
        sys.exit(1)

    # Ensure the output directory exists, create it if not
    if not os.path.isdir(output_directory):
        try:
            os.makedirs(output_directory)
            print(f"Created output directory: '{output_directory}'")
        except OSError as e:
            print(f"Error creating output directory '{output_directory}': {e}", file=sys.stderr)
            sys.exit(1)

    # Find all TXT files in the input directory
    txt_files = glob.glob(os.path.join(input_directory, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{input_directory}'.")
        return

    print(f"Found {len(txt_files)} .txt files in '{input_directory}'. Processing...")

    for txt_path in txt_files:
        print(f"\\nProcessing: {txt_path}")
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
        except Exception as e:
            print(f"Error reading file '{txt_path}': {e}", file=sys.stderr)
            continue

        cleaned_text = clean_text(raw_text)

        base_filename = os.path.basename(txt_path)
        cleaned_txt_path = os.path.join(output_directory, base_filename)

        try:
            with open(cleaned_txt_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
            print(f"Successfully cleaned text and saved to: {cleaned_txt_path}")
        except Exception as e:
            print(f"Error writing cleaned file '{cleaned_txt_path}': {e}", file=sys.stderr)

    print("\\nText cleaning complete.")

if __name__ == "__main__":
    main()
