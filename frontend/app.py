import os

import requests
import streamlit as st

# Configuration
FASTAPI_URL = os.getenv('FASTAPI_URL', 'http://localhost:8000')

st.title('PDF/Webpage Processor')

# File upload Section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
url_input = st.text_input("Or enter webpage URL: ")

if st.button('Process'):
    if uploaded_file:
        # Process PDF
        files = {"file:": (uploaded_file.name, uploaded_file)}
        response = requests.post(
            f"{FASTAPI_URL}/processpdf/",
            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        )

        if response.status_code == 200:
            data = response.content
            st.success("PDF processed successfully!")
            st.download_button(label="Download Markdown", data=data, file_name=f"{uploaded_file.name}.md",
                               mime="application/octet-stream")
        else:
            st.error(f"Error processing PDF: {response.text}")

    elif url_input:
        # Process Webpage
        response = requests.post(
            f"{FASTAPI_URL}/processurl/",
            json={"url": url_input}
        )

        if response.status_code == 200:
            data = response.content
            st.success("Webpage processed successfully!")
            st.download_button(label="Download Markdown", data=data, file_name=f"webpage.md",
                               mime="application/octet-stream")
        else:
            st.error(f"Error processing webpage: {response.text}")

    else:
        st.warning("Please upload a PDF file or enter a webpage URL.")
