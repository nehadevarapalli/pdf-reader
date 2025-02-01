import os
import shutil
from pathlib import Path

from cloud_ops import download_file_from_s3, upload_file_to_s3, upload_directory_to_s3
from webscraper import WebScraper
from docling.document_converter import DocumentConverter
from markitdown import MarkItDown
import uuid
from python_pdf_extraction import extract_text_with_docling, extract_images_to_folder, extract_tables_with_docling

base_dir = Path('./temp_processing')
output = base_dir / Path('output')
s3_bucket = 'neu-pdf-webpage-parser'
s3_pdf_input_prefix = 'pdfs/raw'
s3_html_input_prefix = 'html/raw'

def html_to_md_docling(url: str, job_name: uuid):
    s3_prefix_text = 'pdfs/html-parser/extracted-text'
    s3_prefix_images = 'pdfs/html-parser/extracted-images'
    s3_prefix_tables = 'pdfs/html-parser/extracted-tables'

    # Step 1: Extract images & tables
    out = WebScraper(url, job_name).extract_all()
    html_path = output / 'html' / f'{job_name}.html'

    # Step 2: Upload input HTML to S3
    input_html_s3_key = f'{s3_html_input_prefix}/{job_name}.pdf'
    upload_file_to_s3(str(html_path), input_html_s3_key, bucket_name=s3_bucket)

    # Step 3: Upload images and tables to S3
    images_local_folder = output / "extracted_images"
    tables_local_folder = output / "extracted_tables"
    upload_directory_to_s3(str(images_local_folder), s3_prefix_images, bucket_name=s3_bucket)
    upload_directory_to_s3(str(tables_local_folder), s3_prefix_tables, bucket_name=s3_bucket)

    # Step 4: Extract text and upload Markdown file to S3
    doc_converter = DocumentConverter()
    conv_result = doc_converter.convert(output / 'html' / f'{job_name}.html')
    markdown_output = conv_result.document.export_to_markdown()
    os.makedirs(output / 'markdown', exist_ok=True)
    markdown_path = output / 'markdown' / f'{job_name}.md'
    with open(markdown_path, 'w') as f:
        f.write(markdown_output)
    markdown_s3_key = f'{s3_prefix_text}/{job_name}.md'
    upload_file_to_s3(str(markdown_path), markdown_s3_key, bucket_name=s3_bucket)
    return markdown_path

def html_to_md_markitdown(url: str, job_name: uuid):
    s3_prefix_text = 'pdfs/html-parser/extracted-text'
    s3_prefix_images = 'pdfs/html-parser/extracted-images'
    s3_prefix_tables = 'pdfs/html-parser/extracted-tables'

    # Step 1: Extract images & tables
    out = WebScraper(url, job_name).extract_all()
    html_path = output / 'html' / f'{job_name}.html'

    # Step 2: Upload input HTML to S3
    input_html_s3_key = f'{s3_html_input_prefix}/{job_name}.pdf'
    upload_file_to_s3(str(html_path), input_html_s3_key, bucket_name=s3_bucket)

    # Step 3: Upload images and tables to S3
    images_local_folder = output / "extracted_images"
    tables_local_folder = output / "extracted_tables"
    upload_directory_to_s3(str(images_local_folder), s3_prefix_images, bucket_name=s3_bucket)
    upload_directory_to_s3(str(tables_local_folder), s3_prefix_tables, bucket_name=s3_bucket)

    # Step 4: Extract text and upload Markdown file to S3
    md = MarkItDown()
    markdown_output = md.convert(str(output / 'html' / f'{job_name}.html'))
    os.makedirs(output / 'markdown', exist_ok=True)
    markdown_path = output / 'markdown' / f'{job_name}.md'
    with open(markdown_path, 'w') as f:
        f.write(markdown_output.text_content)
    markdown_s3_key = f'{s3_prefix_text}/{job_name}.md'
    upload_file_to_s3(str(markdown_path), markdown_s3_key, bucket_name=s3_bucket)
    return markdown_path

def standardize_docling(file: Path, job_name: uuid):
    s3_prefix_text = 'pdfs/docling/extracted-text'
    doc_converter = DocumentConverter()
    conv_result = doc_converter.convert(file)
    markdown_output = conv_result.document.export_to_markdown()
    markdown_dir = output / 'markdown'
    os.makedirs(markdown_dir, exist_ok=True)
    markdown_path = markdown_dir / f'{job_name}.md'
    with open(markdown_path, 'w') as f:
        f.write(markdown_output)
    # Upload MD to S3
    markdown_s3_key = f'{s3_prefix_text}/{job_name}.md'
    upload_file_to_s3(str(markdown_path), markdown_s3_key, bucket_name=s3_bucket)
    return markdown_path

def standardize_markitdown(file: Path, job_name: uuid):
    s3_prefix_text = 'pdfs/markitdown/extracted-text'
    md = MarkItDown()
    conv_result = md.convert(str(file))
    markdown_dir = output / 'markdown'
    os.makedirs(markdown_dir, exist_ok=True)
    markdown_path = markdown_dir / f'{job_name}.md'
    with open(markdown_path, 'w') as f:
        f.write(conv_result.text_content)
    # Upload MD to S3
    markdown_s3_key = f'{s3_prefix_text}/{job_name}.md'
    upload_file_to_s3(str(markdown_path), markdown_s3_key, bucket_name=s3_bucket)
    return markdown_path

def pdf_to_md_docling(file: Path, job_name: uuid):
    s3_prefix_text = 'pdfs/python-parser/extracted-text'
    s3_prefix_images = 'pdfs/python-parser/extracted-images'
    s3_prefix_tables = 'pdfs/python-parser/extracted-tables'

    # clean_temp_files()

    # Step 1: Upload input PDF to S3
    input_pdf_s3_key = f'{s3_pdf_input_prefix}/{job_name}.pdf'
    upload_file_to_s3(str(file), input_pdf_s3_key, bucket_name=s3_bucket)

    # Step 2: Extract text and upload Markdown file to S3
    markdown_local_path = output / f'{job_name}.md'
    extract_text_with_docling(file, markdown_local_path)
    markdown_s3_key = f'{s3_prefix_text}/{Path(markdown_local_path).name}'
    upload_file_to_s3(str(markdown_local_path), markdown_s3_key, bucket_name=s3_bucket)

    # Step 3: Extract images and upload the directory to S3
    images_local_folder = output / "extracted_images"
    extract_images_to_folder(file, images_local_folder)
    upload_directory_to_s3(str(images_local_folder), s3_prefix_images, bucket_name=s3_bucket)

    # Step 4: Extract tables and upload the directory to S3
    tables_local_folder = output / "extracted_tables"
    extract_tables_with_docling(file, tables_local_folder)
    upload_directory_to_s3(str(tables_local_folder), s3_prefix_tables, bucket_name=s3_bucket)

    return markdown_local_path

def get_job_name():
    return uuid.uuid4()

def clean_temp_files():
    shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(output, exist_ok=True)

