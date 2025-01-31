import os
import io
import csv
import pdfplumber
import fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image
#from pypdf import PdfReader

# Set output directories
OUTPUT_DIR = "output"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
MD_FILE = os.path.join(OUTPUT_DIR, "output.md")

# Create directories if not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to open a PDF file
def open_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            return io.BytesIO(f.read())
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Function to extract text and tables while maintaining structure
def extract_text_and_tables(pdf_file):
    markdown_content = []
    table_files = []
    
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                markdown_content.append(f"\n## Page {page_num}\n")
                markdown_content.append(page_text.replace("\n", "  \n"))  # Preserve new lines in Markdown
            
            tables_on_page = page.extract_tables()
            if tables_on_page:
                csv_filename = os.path.join(TABLE_DIR, f"table_page_{page_num}.csv")
                table_files.append(csv_filename)
                with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for table in tables_on_page:
                        for row in table:
                            writer.writerow(row)
                markdown_content.append(f"\n### Table on Page {page_num}\n[Table Data](tables/table_page_{page_num}.csv)\n")

    return "\n".join(markdown_content), table_files

# Function to extract images and preserve their order
def extract_images(pdf_file):
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    image_files = []
    markdown_references = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for img_idx, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            image = Image.open(io.BytesIO(img_bytes))
            img_filename = os.path.join(IMAGE_DIR, f"image_page_{page_num + 1}_{img_idx}.jpeg")
            image.save(img_filename, "JPEG")
            image_files.append(img_filename)
            markdown_references.append(f"\n### Image on Page {page_num + 1}\n![Image](images/{os.path.basename(img_filename)})\n")

    return image_files, markdown_references

# Function to extract formulas and preserve structure
def extract_formulas(pdf_file):
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    formula_images = []
    markdown_references = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)

        for img_idx, img in enumerate(images, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            image = Image.open(io.BytesIO(img_bytes))

            # Convert to grayscale and process for OCR
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            _, binarized = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#             # Extract text using OCR
            formula_text = pytesseract.image_to_string(binarized, config="--psm 6")

            if formula_text.strip():
                formula_img_path = os.path.join(IMAGE_DIR, f"formula_page_{page_num + 1}_{img_idx}.jpeg")
                image.save(formula_img_path, "JPEG")
                formula_images.append(formula_img_path)
                markdown_references.append(f"\n### Formula on Page {page_num + 1}\n![Formula](images/{os.path.basename(formula_img_path)})\n")

    return formula_images, markdown_references

# Function to generate Markdown file with preserved structure
def generate_markdown(text, tables, images, formulas):
    with open(MD_FILE, "w", encoding="utf-8") as md:
        md.write("# Extracted PDF Content\n\n")
        md.write(text + "\n\n")

        if tables:
            md.write("\n## Extracted Tables\n")
            for table in tables:
                table_name = os.path.basename(table)
                md.write(f"- [Table: {table_name}](tables/{table_name})\n")

        if images:
            md.write("\n## Extracted Images\n")
            md.writelines(images)

        if formulas:
            md.write("\n## Extracted Formulas\n")
            md.writelines(formulas)

# Main function to process the PDF
def process_pdf(pdf_path):
    pdf_file = open_pdf(pdf_path)
    if not pdf_file:
        return

    print("Extracting text and tables...")
    text, tables = extract_text_and_tables(pdf_file)

    print("Extracting images...")
    images, image_references = extract_images(pdf_file)

    print("Extracting formulas using OCR...")
    formulas, formula_references = extract_formulas(pdf_file)

    print("Generating Markdown output...")
    generate_markdown(text, tables, image_references, formula_references)

    print("Extraction complete! Check the 'output' folder.")

# Run the script with your PDF file path
pdf_file_path = "C:/Users/DELL/Desktop/Bigdata_class/pdf-reader-main/example.pdf"
process_pdf(pdf_file_path)
