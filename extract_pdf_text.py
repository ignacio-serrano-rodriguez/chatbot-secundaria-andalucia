# Creates the python script extract_pdf_text.py
import argparse # Keep argparse for potential future single-file use, but don't use it by default
import fitz  # type: ignore # PyMuPDF
import sys
import os
import glob # Import glob for finding files

def extract_text_from_pdf(pdf_path):
    """
    Extracts raw text content from a PDF file using PyMuPDF.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content, or None if an error occurs.
    """
    try:
        # Open the PDF file
        document = fitz.open(pdf_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{pdf_path}'", file=sys.stderr)
        return None
    except fitz.FitzError as e:
        print(f"Error opening or parsing PDF '{pdf_path}': {e}", file=sys.stderr)
        # This might catch issues like password-protected files if they aren't handled gracefully
        return None
    except Exception as e:
        print(f"An unexpected error occurred while opening '{pdf_path}': {e}", file=sys.stderr)
        return None

    full_text = ""
    try:
        # Iterate through each page
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            # Extract text from the page
            page_text = page.get_text("text") # Use "text" for plain text extraction
            if page_text:
                full_text += page_text + "\n" # Add a newline between pages for readability
    except Exception as e:
        print(f"An error occurred during text extraction from page {page_num + 1} of '{pdf_path}': {e}", file=sys.stderr)
        document.close()
        return None # Return None or partially extracted text based on desired behavior

    # Close the document
    document.close()
    return full_text

def main():
    """
    Main function to find PDF files in the 'PDFs' directory,
    extract text, and save to corresponding .txt files.
    """
    pdf_directory = "PDFs"
    output_directory = "TXTs" # Save .txt files in the TXTs directory

    # Ensure the PDF directory exists
    if not os.path.isdir(pdf_directory):
        print(f"Error: Directory '{pdf_directory}' not found.", file=sys.stderr)
        sys.exit(1)

    # Ensure the output directory exists, create it if not
    if not os.path.isdir(output_directory):
        try:
            os.makedirs(output_directory)
            print(f"Created output directory: '{output_directory}'")
        except OSError as e:
            print(f"Error creating output directory '{output_directory}': {e}", file=sys.stderr)
            sys.exit(1)

    # Find all PDF files in the specified directory
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{pdf_directory}'.")
        return

    print(f"Found {len(pdf_files)} PDF files in '{pdf_directory}'. Processing...")

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path}")
        extracted_text = extract_text_from_pdf(pdf_path)

        if extracted_text is not None:
            # Construct the output file path by replacing .pdf with .txt
            base_filename = os.path.basename(pdf_path)
            txt_filename = os.path.splitext(base_filename)[0] + ".txt"
            txt_path = os.path.join(output_directory, txt_filename)

            try:
                # Write the extracted text to the .txt file with UTF-8 encoding
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(extracted_text)
                print(f"Successfully extracted text and saved to: {txt_path}")
            except IOError as e:
                print(f"Error writing to file '{txt_path}': {e}", file=sys.stderr)
            except Exception as e:
                 print(f"An unexpected error occurred while writing '{txt_path}': {e}", file=sys.stderr)
        else:
            print(f"Failed to extract text from: {pdf_path}", file=sys.stderr)

    print("\nProcessing complete.")


if __name__ == "__main__":
    # Before running, ensure PyMuPDF is installed:
    # pip install PyMuPDF
    main()