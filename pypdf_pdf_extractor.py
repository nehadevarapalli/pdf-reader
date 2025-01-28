import os
from pypdf import PdfReader
import camelot


pdf_file = './example.pdf'
output_file = 'pypdf_extracted_output.txt'
image_folder = "pypdf_extracted_images"
table_folder = "pypdf_extracted_tables"
reader = PdfReader(pdf_file)

# Extract text from the PDF
def extract_text_from_pdf(reader, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                f.write(f'---- Page {page_number + 1} ----\n')
                f.write(text + '\n')
            else:
                f.write(f'---- Page {page_number + 1} ----\n')
                f.write('No text found on this page.\n')   
    print(f'Text extraction completed from {pdf_file} and saved to {output_file}')

# Extract images from the PDF
def extract_images_from_pdf(reader, output_file, image_folder):
    os.makedirs(image_folder, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        for page_number, page in enumerate(reader.pages):
                for image_index, image_file_object in enumerate(page.images):
                    # Generate unique image filename
                    image_name = f"page_{page_number + 1}_image_{image_index + 1}_{image_file_object.name}"
                    image_path = os.path.join(image_folder, image_name)

                    # Save the image to the folder
                    with open(image_path, "wb") as image_file:
                        image_file.write(image_file_object.data)

                    # Log the saved image details in the output file
                    f.write(f"Saved: {image_path}\n")
    print(f"Image extraction completed. Images saved to '{image_folder}'.")

# Extract tables from the PDF
def extract_tables_from_pdf(pdf_file, output_file, table_folder):
    os.makedirs(table_folder, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        try:
            # Extract tables using Camelot
            tables = camelot.read_pdf(pdf_file, pages="all", flavor="lattice") # Try flavor="stream" later

            # Saving each table as a CSV file
            for i, table in enumerate(tables):
                table_name = f"table_{i + 1}.csv"
                table_path = os.path.join(table_folder, table_name)
                table.to_csv(table_path)

            print(f"Table extraction completed. Tables saved to '{table_folder}'.")
        except Exception as e:
            print(f"Error extracting tables: {e}")

# Main function to perform complete extraction
def main():
    # Extract text and save to file
    extract_text_from_pdf(reader, output_file)

    # Extract images and save to folder
    extract_images_from_pdf(reader, output_file, image_folder)

    # Extract tables and save to folder
    extract_tables_from_pdf(pdf_file, output_file, table_folder)

if __name__ == '__main__':
    main()