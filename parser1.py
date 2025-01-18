from pypdf import PdfReader

reader = PdfReader("example.pdf")

text = []

for page in reader.pages:
    print(page.extract_text())
    for count, image_file_object in enumerate(page.images):
        with open('extracted/'+str(count) + image_file_object.name, "wb") as fp:
            fp.write(image_file_object.data)