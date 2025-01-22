# pdf-reader

# Python PDF Parser

This Python tool extracts text, tables, images, and potential charts from PDF files. It uses a combination of libraries such as `pdfplumber`, `PyMuPDF (fitz)`, and `PIL` to achieve these functionalities.

## Setup Instructions

### Recommended IDE: Visual Studio Code (VSCode)

To use the tool, we recommend Visual Studio Code for its great support for Python and ease of use. Follow the steps below to set up your environment.

### Step 1: Clone or Download the Repository

1. **Clone the repository** (or download the code if you're working with local files):
    ```bash
    git clone https://github.com/your-repository-path.git
    ```

2. **Open the project directory in VSCode**:
    ```bash
    cd your-repository-path
    code .
    ```

### Step 2: Install Required Libraries

Ensure you have all required Python libraries installed. The easiest way is to use a `requirements.txt` file, which contains the necessary libraries for the project.

**Install dependencies** using pip:

Open the integrated terminal in VSCode and run the following command:

```bash
pip install -r requirements.txt
```

### Step 3: Running the Script (`parser.py`)


To run the script, execute the following command in your VSCode terminal:

```bash
python parser.py
```

**Results:**

- **Text Extracted**: The total number of characters extracted from the PDF's text.
- **Tables Extracted**: The total number of tables found in the PDF.
- **Found Images**: The total number of images extracted.
- **Potential Charts**: The total number of images that could be charts (this is a heuristic detection).
- **Visualizations**: If any charts are identified, they are visualized and displayed as images using `matplotlib`. This is helpful when the PDF contains graphs or data visualizations that need to be analyzed.

**Findings:**
The table extraction functionality in the tool does not currently work as expected. Instead of extracting tables in their structured format (with rows and columns), the tables are being returned as plain text. 



## Accuracy Assessment

The accuracy of the tool depends on the quality and structure of the PDF. The text extraction is quite reliable in most PDFs, but complex layouts, scanned images, or embedded fonts may reduce its accuracy.



## Processing Time

The time taken for processing will depend on the size and complexity of the PDF. For example, with a PDF containing 75 images and 47,000 characters of text, the processing time was as follows:

- CPU times: user 4.5 s
- sys: 21.4 ms
- total: 4.52 s 
- Wall time: 4.61 s

This indicates that the processing was quite fast (around 4.5 seconds for a PDF of this size).

### Factors Affecting Processing Time
- **PDF Size**: Larger PDFs with more pages, images, or tables will take longer.
- **Image Complexity**: Extracting high-quality images may take additional processing time.
- **Table Structure**: Complex or poorly structured tables might increase extraction time.



## Conclusion

This tool provides a comprehensive and fast way to extract and visualize content from PDFs, including text, tables, images, and charts. It is useful for a variety of applications such as data extraction, document analysis, and report generation.


