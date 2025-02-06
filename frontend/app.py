import os
from io import BytesIO
from zipfile import ZipFile

import streamlit as st
import requests

# Configuration
FASTAPI_URL = os.getenv('FASTAPI_URL', 'https://nehadevarapalli-parseforge.hf.space')
APP_NAME = "ParseForge"
APP_DESCRIPTION = """📄🌐 A versatile document processing tool that converts PDFs and webpages into structured markdown content and extracts all data. 
Choose between our **custom Python parser** (built with PyMuPDF, Docling, and BeautifulSoup) or the **enterprise-grade Llama parser** for comparison."""

st.set_page_config(page_title=APP_NAME, page_icon="⚙️", layout="centered")

if 'input_type' not in st.session_state:
    st.session_state.input_type = 'PDF'

# App Header
st.title(f"⚙️ {APP_NAME}")
st.markdown(f"*{APP_DESCRIPTION}*")
st.divider()

# Input Type Selector
input_col, parser_col = st.columns([1, 1])
with input_col:
    input_type = st.radio("Select Input Type", 
                          ["📄 PDF File", "🌐 Webpage URL"],
                          horizontal=True)

with parser_col:
    parser_type = st.selectbox(
        "Choose Parser Engine:",
        ["Python Parser", "Llama Parser", "Standardize Docling", "Standardize MarkItDown"],
        index=0,  # Default selection
        format_func=lambda x: "Select Parser" if x == "" else x,
        help="""Python Parser: Custom-built using PyMuPDF (image extraction), Docling (text & table extraction), 
BeautifulSoup (webpage parsing). Optimized for specific use cases.\nLlama Parser: AI-powered enterprise solution for superior accuracy.\n Standardize Docling: Standardize document using Docling.\nStandardize MarkItDown: Standardize document using MarkItDown.""",
        disabled=False
    )

# File/URL Input Section
if "PDF" in input_type:
    uploaded_file = st.file_uploader("Upload PDF",
                                     type=["pdf"],
                                     help="Maximum file size: 10MB. Extracts text, tables, and images.")
else:
    url_input = st.text_input("Enter Webpage URL: ",
                                placeholder="https://example.com",
                                help="Paste a webpage URL to process. Extracts text, tables, and images.")

st.divider()

# Output options
if parser_type not in ["Standardize Docling", "Standardize MarkItDown"]:
    st.subheader("🔧 Output Options")
    output_col = st.columns([1])[0]

    with output_col:
        output_formats = st.multiselect(
            "Select components to include:",
            options=["Markdown", "Images", "Tables"],
            default=["Markdown"],
            help="Choose which components to include in your output."
        )

    process_disabled = len(output_formats) == 0 or ("PDF" in input_type and not uploaded_file) or ("Webpage" in input_type and not url_input)

    if process_disabled and len(output_formats) == 0:
        st.caption("ℹ️ Please select at least one output component to enable processing.")
    if process_disabled and ("PDF" in input_type and not uploaded_file or "Webpage" in input_type and not url_input):
        st.caption("ℹ️ Please provide a valid input to enable processing.")
else:
    process_disabled = ("PDF" in input_type and not uploaded_file) or ("Webpage" in input_type and not url_input)

# Process Button
if st.button("✨ Process Content", type="primary", use_container_width=True, disabled=process_disabled):
    if parser_type in ["Standardize Docling", "Standardize MarkItDown"]:
        if "PDF" in input_type and uploaded_file:
            endpoint = "/standardizedoclingpdf/" if parser_type == "Standardize Docling" else "/standardizemarkitdownpdf/"
            with st.spinner("🔍 Standardizing PDF content..."):
                response = requests.post(
                    f"{FASTAPI_URL}{endpoint}",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                )
        else:
            endpoint = "/standardizedoclingurl/" if parser_type == "Standardize Docling" else "/standardizemarkitdownurl/"
            with st.spinner("🌐 Standardizing webpage content..."):
                response = requests.post(
                    f"{FASTAPI_URL}{endpoint}",
                    json={"url": url_input}
                )

        if response.status_code == 200:
            st.success("✅ Standardization completed successfully!")
            st.download_button(
                label="⬇️ Download Standardized Document",
                data=response.content,
                file_name=f"{uploaded_file.name}.md" if "PDF" in input_type else "webpage.md",
                mime="text/markdown"
            )
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")
    else:  
        params = {
            "include_markdown": "Markdown" in output_formats,
            "include_images": "Images" in output_formats,
            "include_tables": "Tables" in output_formats,
        }

        if "PDF" in input_type and uploaded_file:
            with st.spinner("🔍 Parsing PDF content..."):
                response = requests.post(
                    f"{FASTAPI_URL}/processpdf/",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    params=params
                )
        else:
            with st.spinner("🌐 Analyzing webpage content..."):
                response = requests.post(
                    f"{FASTAPI_URL}/processurl/",
                    json={"url": url_input},
                    params=params
                )

        if response.status_code == 200:
            content_disposition = response.headers.get("Content-Disposition")
            if "attachment; filename=" in content_disposition:
                filename = content_disposition.split("filename=")[1]
                if filename.endswith(".zip"):
                    st.success("✅ All components processed successfully!")
                    st.download_button(
                        label="⬇️ Download ZIP Archive",
                        data=response.content,
                        file_name=filename,
                        mime="application/zip"
                    )
                else:
                    st.success("✅ Markdown processed successfully!")
                    st.download_button(
                        label="⬇️ Download Markdown",
                        data=response.content,
                        file_name=f"{uploaded_file.name}.md",
                        mime="text/markdown"
                    )
            else:
                st.error("Unexpected response format. Please try again.")
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")
    
# Feature Explanation
with st.expander("ℹ️ About ParseForge Features"):
    st.markdown("""
    **Key Features:**
    - **Custom Python Parser** *(Default Option)*:
      - **PDF Parsing:** 
        - Built using PyMuPDF for extracting images from PDFs.
        - Uses Docling to extract text and tables with precision.
      - **Webpage Parsing:** 
        - Powered by BeautifulSoup for extracting structured content from webpages.
        - Suitable for projects requiring lightweight, rule-based parsing.
    - **Llama Parser** *(Enterprise Option)*:
      - AI-powered solution offering superior accuracy for complex layouts.
      - Ideal for enterprise use cases where advanced document understanding is required.
    - **Standardization Options:**
        - **Standardize Docling:** 
            - Standardize document structure using Docling.
        - **Standardize MarkItDown:** 
            - Standardize document structure using MarkItDown.
    - **Multi-Format Support:**
      - PDF documents (scanned & digital)
      - Webpages (articles, blogs, documentation)
    - **Output Options:**
      - Clean Markdown formatting
      - Preserved document structure
    """)