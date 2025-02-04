import streamlit as st
import requests
import os

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
        ["Python Parser", "Llama Parser"],
        index=0,  # Default selection
        format_func=lambda x: "Select Parser" if x == "" else x,
        help="""Python Parser: Custom-built using PyMuPDF (image extraction), Docling (text & table extraction), 
BeautifulSoup (webpage parsing). Optimized for specific use cases.\nLlama Parser: AI-powered enterprise solution for superior accuracy.""",
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
st.subheader("🔧 Output Options")
output_col1, output_col2 = st.columns([1, 1])

with output_col1:
    output_formats = st.multiselect(
        "Select components to include:",
        options=["Markdown", "Images", "Tables"],
        default=["Markdown"],
        help="Choose which components to include in your output."
    )

with output_col2:
    bundle_files = st.checkbox(
        "📦 Bundle selected components in ZIP",
        value=True,
        help="Combine all selected formats into a single downloadable archive"
    )

if not bundle_files and (len(output_formats) > 1):
    st.warning("⚠️ Multiple components require bundling to download!")
    st.stop()

# Process Button
if st.button("✨ Process Content", type="primary", use_container_width=True):
    params = {
        "include_markdown": "Markdown" in output_formats,
        "include_images": "Images" in output_formats,
        "include_tables": "Tables" in output_formats,
        "bundle": bundle_files
    }

    if "PDF" in input_type and uploaded_file:
        with st.spinner("🔍 Parsing PDF content..."):
            response = requests.post(
                f"{FASTAPI_URL}/processpdf/",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                params=params
            )

        if response.status_code == 200:
            if bundle_files:
                sucess_msg = st.success("✅ All components bundled successfully!")
                st.download_button (
                    label="⬇️ Download ZIP Archive",
                    data=response.content,
                    file_name=f"{uploaded_file.name}_bundle.zip",
                    mime="application/zip"
                )
            else:
                success_msg = st.success("✅ PDF processed successfully!"),
                st.download_button(
                    label="⬇️ Download Markdown",
                    data=response.content,
                    file_name=f"{uploaded_file.name}.md",
                    mime="text/markdown"
                )
        else:
            st.error(f"❌ Processing failed: {response.text}")
        
    elif "Webpage" in input_type and url_input:
        with st.spinner("🌐 Analyzing webpage content..."):
            response = requests.post(
                f"{FASTAPI_URL}/processurl/",
                json={"url": url_input},
                params=params
            )
            
        if response.status_code == 200:
            if bundle_files:
                success_msg = st.success("✅ All components bundled successfully!")
                st.download_button(
                    label="⬇️ Download ZIP Archive",
                    data=response.content,
                    file_name="webpage_bundle.zip",
                    mime="application/zip"
                )
            else:
                success_msg = st.success("✅ Webpage processed successfully!")
                st.download_button(
                    label="⬇️ Download Markdown",
                    data=response.content,
                    file_name="webpage.md",
                    mime="text/markdown"
                )
        else:
            st.error(f"❌ Processing failed: {response.text}")
    
    else:
        st.warning("⚠️ Please provide a valid input first!")

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
    - **Multi-Format Support:**
      - PDF documents (scanned & digital)
      - Webpages (articles, blogs, documentation)
    - **Output Options:**
      - Clean Markdown formatting
      - Preserved document structure
    """)