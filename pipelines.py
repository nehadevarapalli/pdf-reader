import os
from pathlib import Path
from webscraper import WebScraper
from docling.document_converter import DocumentConverter
from markitdown import MarkItDown
import uuid

output = Path('./output')

def html_to_md_docling(url: str, job_name: uuid):
    doc_converter = DocumentConverter()
    out = WebScraper(url, job_name).extract_all()
    conv_result = doc_converter.convert(output / 'html' / f'{job_name}.html')
    markdown_output = conv_result.document.export_to_markdown()
    os.makedirs(output / 'markdown', exist_ok=True)
    markdown_path = output / 'markdown' / f'{job_name}.md'
    with open(markdown_path, 'w') as f:
        f.write(markdown_output)
    return markdown_path

def html_to_md_markitdown(url: str, job_name: uuid):
    md = MarkItDown()
    out = WebScraper(url, job_name).extract_all()
    markdown_output = md.convert(str(output / 'html' / f'{job_name}.html'))
    os.makedirs(output / 'markdown', exist_ok=True)
    markdown_path = output / 'markdown' / f'{job_name}.md'
    with open(markdown_path, 'w') as f:
        f.write(markdown_output.text_content)
    return markdown_path

def standardize_docling(file: Path, job_name: uuid):
    doc_converter = DocumentConverter()
    conv_result = doc_converter.convert(file)
    markdown_output = conv_result.document.export_to_markdown()
    markdown_dir = output / 'markdown'
    os.makedirs(markdown_dir, exist_ok=True)
    markdown_path = markdown_dir / f'{job_name}.md'
    with open(markdown_path, 'w') as f:
        f.write(markdown_output)
    return markdown_path

def standardize_markitdown(file: Path, job_name: uuid):
    md = MarkItDown()
    conv_result = md.convert(str(file))
    markdown_dir = output / 'markdown'
    os.makedirs(markdown_dir, exist_ok=True)
    markdown_path = markdown_dir / f'{job_name}.md'
    with open(markdown_path, 'w') as f:
        f.write(conv_result.text_content)
    return markdown_path

def get_job_name():
    return uuid.uuid4()
