import os
import fitz
import pandas as pd
from pathlib import Path
from docling.document_converter import DocumentConverter
from cloud_ops import (
    download_file_from_s3,
    upload_file_to_s3,
    upload_directory_to_s3,
    add_tags_to_object,
)

# Input and output configurations
s3_bucket = 'neu-pdf-webpage-parser'
s3_input_prefix = 'pdfs/raw'
s3_prefix_text = 'pdfs/python-parser/extracted-text'
s3_prefix_images = 'pdfs/python-parser/extracted-images'
s3_prefix_tables = 'pdfs/python-parser/extracted-tables'

# Local processing directories (volatile storage)
local_base_dir = Path('./temp_processing') 
output_dir = local_base_dir / Path('output')

# Extract text from the PDF and write to Markdown
def extract_text_with_docling(pdf_file, markdown_file):
    converter = DocumentConverter()
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
            xref = img[0]  
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]  
            image_name = f'{Path(pdf_file).stem}_page_{page_number + 1}_image_{img_index + 1}.{image_ext}'
            image_path = os.path.join(image_folder, image_name)

            with open(image_path, 'wb') as image_file:
                image_file.write(image_bytes)

    print(f'Images extracted and saved to "{image_folder}".')

# Extract tables from the PDF and save them to a folder
def extract_tables_with_docling(pdf_file, table_folder):
    converter = DocumentConverter()
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
    os.makedirs(local_base_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Download input PDF from S3
    input_pdf_s3_key = f'{s3_input_prefix}/example.pdf'
    local_pdf_path = download_file_from_s3(input_pdf_s3_key, str(local_base_dir), bucket_name=s3_bucket)

    if not local_pdf_path:
        print(f"Error downloading PDF from '{input_pdf_s3_key}'.")
        return
    
    # Step 2: Extract text and upload Markdown file to S3
    markdown_local_path = output_dir / f'{Path(local_pdf_path).stem}_extracted_output.md'
    extract_text_with_docling(local_pdf_path, markdown_local_path)
    markdown_s3_key = f'{s3_prefix_text}/{Path(markdown_local_path).name}'
    upload_file_to_s3(str(markdown_local_path), markdown_s3_key, bucket_name=s3_bucket)

    # Step 3: Extract images and upload the directory to S3
    images_local_folder = output_dir / "extracted_images"
    extract_images_to_folder(local_pdf_path, images_local_folder)
    upload_directory_to_s3(str(images_local_folder), s3_prefix_images, bucket_name=s3_bucket)

    # Step 4: Extract tables and upload the directory to S3
    tables_local_folder = output_dir / "extracted_tables"
    extract_tables_with_docling(local_pdf_path, tables_local_folder)
    upload_directory_to_s3(str(tables_local_folder), s3_prefix_tables, bucket_name=s3_bucket)

if __name__ == '__main__':
    main()
