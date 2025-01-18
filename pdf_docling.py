from docling.document_converter import DocumentConverter

source = "./2105.04895v1.pdf"
converter = DocumentConverter()
result = converter.convert(source)

# Print results to the console
print(result.document.export_to_text())

# Saving results to a text file
with open('docling_conversion_results.txt', 'w', encoding='utf-8') as f:
    f.write("\n\nMarkdown Conversion:\n")
    f.write(result.document.export_to_markdown())