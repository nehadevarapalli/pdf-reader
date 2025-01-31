import os
import fitz
import pandas as pd
from pathlib import Path
from docling.document_converter import DocumentConverter

# Input PDF file
pdf_file = Path("./example.pdf")

# Output files and folders
output_dir = Path("./python_pdf_extraction_output")
markdown_file = output_dir / Path("extracted_output.md")  
image_folder = output_dir / Path("extracted_images")
table_folder = output_dir / Path("extracted_tables")

converter = DocumentConverter()

# Extract text from the PDF and write to Markdown
def extract_text_with_docling(pdf_file, markdown_file):
    result = converter.convert(pdf_file)

    markdown_content = result.document.export_to_markdown()

    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f'Text extracted and saved to "{markdown_file}".')

# Extract images from the PDF and save them to a folder
def extract_images_to_folder(pdf_file, image_folder):
    os.makedirs(image_folder, exist_ok=True)
    doc = fitz.open(pdf_file) 

    for page_number in range(len(doc)):
        page = doc[page_number]

        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]  # XREF of the image
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]  
            image_name = f'page_{page_number + 1}_image_{img_index + 1}.{image_ext}'
            image_path = os.path.join(image_folder, image_name)

            with open(image_path, 'wb') as image_file:
                image_file.write(image_bytes)

    print(f'Images extracted and saved to "{image_folder}".')

# Extract tables from the PDF and save them to a folder
def extract_tables_with_docling(pdf_file, table_folder):
    conv_res = converter.convert(pdf_file)
    os.makedirs(table_folder, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    for table_ix, table in enumerate(conv_res.document.tables):
        table_df: pd.DataFrame = table.export_to_dataframe()

        element_csv_filename = table_folder / f'{doc_filename}-table-{table_ix+1}.csv'
        print(f'Saving CSV table to {element_csv_filename}')
        table_df.to_csv(element_csv_filename)

    print(f"Tables extracted and saved to '{table_folder}'.")

# Main function to perform complete extraction
def main():
    os.makedirs(output_dir, exist_ok=True)

    # Extract text with Docling
    extract_text_with_docling(pdf_file, markdown_file)

    # Extract images to a folder using pymupdf
    extract_images_to_folder(pdf_file, image_folder)

    # Extract tables with Docling
    extract_tables_with_docling(pdf_file, table_folder)

if __name__ == '__main__':
    main()
