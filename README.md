# **ParseForge: Automating Text Extraction from PDFs and URLs**

## **Introduction**

This project demonstrates the functionality of a context extraction tool that extracts structured information from unstructured data sources like PDFs and web pages. The tool allows users to test and compare the performance of **open-source** and **enterprise-grade parsers**, providing insights into their efficiency, accuracy, and feasibility.

### **Key Features**
- **Dual Processing Modes**:
  - 🐍 Open-Source Stack (PyMuPDF, BeautifulSoup, Docling).
  - 🚀 Enterprise Solutions (LlamaParser, Firecrawl).
- **Multi-Format Support**: Extracts text, images, tables, and metadata from PDFs and web pages.
- **Smart Output Options**: Choose between Markdown-only or bundled ZIP files containing multiple components.
- **Cloud Integration**: Uses AWS S3 for secure storage of raw inputs and processed outputs.

---

## **Initial Setup**

### **Prerequisites**
1. Install Python (>= 3.8) on your system.
2. Install Docker (for containerized deployment).
3. Set up an AWS account for S3 storage (if running locally).

### **Installation**
1. Clone the repository:
```
git clone https://github.com/your-repo/parse-forge.git
cd parse-forge
```

2. Install dependencies:
```
pip install -r requirements.txt
```


3. Set up environment variables by creating a `.env` file in the root directory (see sample below).

4. Run the application:
- **Frontend (Streamlit):**
  ```
  streamlit run app.py
  ```
- **Backend (FastAPI):**
  ```
  uvicorn api:app --reload
  ```

5. Access the application:
- Streamlit Frontend: `http://localhost:8501`
- FastAPI Backend: `http://localhost:8000/docs`

---

## **Sample `.env` File**

Create a `.env` file in the root directory with the following structure:
AWS Configuration
```
AWS_ACCESS_KEY_ID=<your_aws_access_key>
AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>
AWS_REGION=<your_aws_region>
S3_BUCKET_NAME=<your_s3_bucket_name>
```
FastAPI Configuration
```
FASTAPI_URL=<link_to_FASTAPI>
```
Enterprise API Keys
```
LLAMAPARSER_API_KEY=<your_llamaparser_api_key>
FIRECRAWL_API_KEY=<your_firecrawl_api_key>
```

Replace `<your_aws_access_key>` and other placeholders with your actual credentials.

---

## **Links**

### Assignment 1 Links:
- **FastAPI Backend**: [https://nehadevarapalli-parseforge.hf.space/](https://nehadevarapalli-parseforge.hf.space/)  
- **Streamlit Frontend**: [https://parse-forge.streamlit.app](https://parse-forge.streamlit.app)  
- **GitHub Project**: [https://github.com/users/nehadevarapalli/projects/2](https://github.com/users/nehadevarapalli/projects/2)  
- **Codelabs Documentation**: [https://codelabs-preview.appspot.com/?file_id=1SZHxAEETpt6-INannVHcy-WhCiZ-rmFsuChKF19sKO8#0](https://codelabs-preview.appspot.com/?file_id=1SZHxAEETpt6-INannVHcy-WhCiZ-rmFsuChKF19sKO8#0)
- **Demo Video**: [Youtube](https://www.youtube.com/watch?v=nrw8KiRCwU4)

---

## **How It Works**

1. Upload a PDF or input a webpage URL.
2. Choose between open-source or enterprise parsers.
3. Select specific components to extract (e.g., text, images, tables).
4. Process the content and download the output as Markdown or a ZIP bundle.

