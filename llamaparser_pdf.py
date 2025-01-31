import os
import base64
import logging
from dotenv import load_dotenv
from llama_parse import LlamaParse
from pdf2image import convert_from_path  # Fallback for image extraction

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Get API key and PDF path from environment variables
LLAMA_API_KEY = os.getenv('LLAMAPARSE_API_KEY')
pdf_path = os.getenv('new_pdf')

# Validate environment variables
if not LLAMA_API_KEY:
    raise ValueError("LLAMAPARSE_API_KEY not found in environment variables")
if not pdf_path:
    raise ValueError("PDF file path not found in environment variables")

# Create output directory for images
output_dir = os.path.join(os.getcwd(), 'extracted_images')
os.makedirs(output_dir, exist_ok=True)

# Initialize LlamaParse with updated parameters
parser = LlamaParse(
    api_key=LLAMA_API_KEY,
    result_type="markdown",  # Using "markdown" since "structured" and "json" fail
    verbose=True,
    language="en",
    auto_mode_trigger_on_image_in_page=True,
    auto_mode_trigger_on_table_in_page=True,
    complemental_formatting_instruction="Extract and preserve all images",
    content_guideline_instruction="Include all visual elements",
    disable_image_extraction=False,
    take_screenshot=True
)

try:
    # Load and parse the PDF
    logging.info(f"Starting to parse file: {pdf_path}")
    documents = parser.load_data(pdf_path)
    logging.info(f"File parsing completed. Raw response: {documents}")

    # Debug: Check document contents
    for doc in documents:
        logging.info(f"Document metadata: {doc.metadata}")

    # Counter for image naming
    image_count = 0
    extracted_images = False

    # Extract and save images from LlamaParse
    for doc in documents:
        logging.info(f"Checking for images in parsed document.")

        # Check `image_resource` (Primary method)
        if hasattr(doc, 'image_resource') and doc.image_resource:
            logging.info(f"Number of images found in `image_resource`: {len(doc.image_resource)}")
            for image_data in doc.image_resource:
                try:
                    image_bytes = base64.b64decode(image_data)
                    image_filename = f'extracted_image_{image_count+1}.png'
                    image_filepath = os.path.join(output_dir, image_filename)

                    with open(image_filepath, 'wb') as img_file:
                        img_file.write(image_bytes)

                    logging.info(f"Saved image: {image_filepath}")
                    image_count += 1
                    extracted_images = True
                except Exception as e:
                    logging.error(f"Error saving image: {e}", exc_info=True)

        # Check `extra_info['images']` (Fallback method)
        elif hasattr(doc, 'extra_info') and isinstance(doc.extra_info, dict) and 'images' in doc.extra_info:
            logging.info(f"Number of images found in `extra_info`: {len(doc.extra_info['images'])}")
            for image_data in doc.extra_info['images']:
                try:
                    image_bytes = base64.b64decode(image_data)
                    image_filename = f'extracted_image_{image_count+1}.png'
                    image_filepath = os.path.join(output_dir, image_filename)

                    with open(image_filepath, 'wb') as img_file:
                        img_file.write(image_bytes)

                    logging.info(f"Saved image: {image_filepath}")
                    image_count += 1
                    extracted_images = True
                except Exception as e:
                    logging.error(f"Error saving image: {e}", exc_info=True)

    # If no images were found, use `pdf2image` 
    if not extracted_images:
        logging.warning("No images extracted from LlamaParse. Using pdf2image as a fallback.")

        # Convert PDF pages to images
        images = convert_from_path(pdf_path)
        for i, image in enumerate(images):
            image_filename = f"fallback_extracted_image_{i+1}.png"
            image_filepath = os.path.join(output_dir, image_filename)
            image.save(image_filepath, "PNG")
            logging.info(f"Saved fallback image: {image_filepath}")

    logging.info(f"\nTotal images extracted: {image_count}")
    logging.info(f"Images saved to: {output_dir}")

except Exception as e:
    logging.error(f"An error occurred during processing: {e}", exc_info=True)

finally:
    logging.info("Processing completed")
