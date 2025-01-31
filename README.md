# doc-reader

# PDF Parsing with Python: Methods & Implementation

## **Overview**
This document outlines the methodologies tested for parsing PDFs in Python, focusing on the optimal combination of **Docling** (for text/table extraction) and **PyMuPDF** (for images). Alternative methods (`PyPDF`, `PDFPlumber`, `Camelot`) were prototyped and archived under `/prototyping`.

---

## **Accuracy Assessment of PDF Parsing Libraries**

| **Library**     | **Text Extraction** | **Image Extraction** | **Table Extraction** | **Speed** | **Complexity** |  
|------------------|----------------------|----------------------|----------------------|-----------|-----------------|
| **PyPDF**        | Good (text-based)   | Basic (no OCR)       | Poor                 | Fast      | Low             |  
| **PyMuPDF**      | Good (text-based)   | **Excellent**        | Poor                 | Fast      | Moderate        |  
| **PDFPlumber**   | Good (text-based)   | Good (no OCR)        | Moderate             | Moderate  | High            |  
| **Camelot**      | None                | None                 | Good (But doesn't detect semi-lattice tables)        | Slow      | High            |  
| **Docling**      | **Excellent**       | Poor                 | **Excellent (Detects all kinds of tables accurately)**    | Moderate  | Moderate        |  

### **Key Observations**
- **Docling**: Best for text (preserves structure) and AI-based table extraction but lacks in image extraction qualities. 
- **PyMuPDF**: Superior for image extraction and fast text extraction.  
- **Camelot**: Specialized for tables but requires Ghostscript on the OS level and Java.  
- **Combination**: **Docling + PyMuPDF** covers text, tables, and images efficiently.

---

## **Optimal Workflow: Docling + PyMuPDF**

### **How They Work Together**
1. **Text & Tables**:
   - **Docling** extracts text and tables into structured Markdown/CSV.  
   - Uses AI models for complex layouts but may miss nested tables.  
2. **Images**:
   - **PyMuPDF** extracts embedded images with metadata (resolution, format).  
   - Saves images as `PNG`/`JPEG` with page-wise naming.

### **Pros**  
- **Accuracy**: Docling preserves formatting; PyMuPDF handles high-res images.  
- **Speed**: Parallel processing possible for large PDFs.  
- **Integration**: Seamless S3 integration for cloud storage.  

### **Cons**  
- **Docling**: Limited to text-based PDFs (no OCR for scanned files).  
- **PyMuPDF**: No native table extraction.  

---

## **Local Setup**

### **Step 1: Create a Virtual Environment**
**For Unix/macOS**
```
python -m venv venv source venv/bin/activate
```
**For Windows**
```
python -m venv venv .\venv\Scripts\activate
```


### **Step 2: Install Dependencies**
```
pip install -r requirements.txt
```

### **Step 3: Configure AWS Credentials**
Run this command and enter the credentials provided:
```
aws configure
```
You'll be prompted for:
```
AWS Access Key ID: [YOUR_ACCESS_KEY]
AWS Secret Access Key: [YOUR_SECRET_KEY]
Default region name: us-east-2  
Default output format: [Press enter for None]
```
**Bucket Structure**:
```
neu-pdf-webpage-parser/
├── pdfs/
│   ├── raw/                   # Raw PDFs
│   ├── python-parser/          # Processed outputs
│       ├── extracted-text/     # Markdown files
│       ├── extracted-images/   # Images (PNG/JPEG)
│       ├── extracted-tables/   # CSV files
```


### **Run the Script**
```
python python_pdf_extraction.py
```


### **Cleanup**
```
deactivate  # Exit the virtual environment
```


This setup ensures isolation of dependencies and reproducibility across environments. The virtual environment (`venv/`) and temporary files (`temp_processing/`) are excluded from Git via `.gitignore`.

---

## **Future Improvements**
- **OCR Integration**: Add Tesseract for scanned PDFs.  
- **Error Handling**: Retry mechanisms for S3 uploads.  
- **Scalability**: Batch processing for multiple PDFs.  

---

## **Conclusion**
The **Docling + PyMuPDF** combination balances accuracy, speed, and ease of integration with cloud workflows. Archived prototypes (`PyPDF`, `PDFPlumber`, `Camelot`) are available in `/prototyping` for reference.
