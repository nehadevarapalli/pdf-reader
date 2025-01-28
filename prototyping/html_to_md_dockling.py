from webscraper import WebScraper
from docling.document_converter import DocumentConverter

doc_converter = DocumentConverter()
out_path = WebScraper('https://www.crummy.com/software/BeautifulSoup/bs4/doc/', 'bs4docs->MD').extract_all()

conv_result = doc_converter.convert(out_path + '/index.html')
# conv_result = doc_converter.convert('https://www.crummy.com/software/BeautifulSoup/bs4/doc/')

# Export to Markdown
markdown_output = conv_result.document.export_to_markdown()
with open(f'{out_path}/output.md', 'w') as f:
    f.write(markdown_output)