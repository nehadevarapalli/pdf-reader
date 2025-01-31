import os
from pathlib import Path
from bs4 import BeautifulSoup, ResultSet
import requests
import pandas as pd
from io import StringIO

class WebScraper:

    def __init__(self, url: str, job_name: str):
        self.url = url
        self.job_name = job_name
        webpage = self.get_webpage()
        self.soup = BeautifulSoup(webpage, 'lxml')
        self.output = Path("./output")
        self.PDF_HOME = self.output / 'pdf'
        self.CSV_HOME = self.output / 'csv'
        self.HTML_HOME = self.output / 'html'
        self.IMG_HOME = self.output / 'images'
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.PDF_HOME, exist_ok=True)
        os.makedirs(self.CSV_HOME, exist_ok=True)
        os.makedirs(self.HTML_HOME, exist_ok=True)
        os.makedirs(self.IMG_HOME, exist_ok=True)
        self.file_outputs = {'image_count': 0 , 'table_count': 0}

    def extract_all(self) -> dict[str, int]:
        """
        Extracts all tables and images from input html
        :return: {image_count: int, table_count: int, home: Path}
        """
        tables, images = self.soup.find_all('table'), self.soup.find_all('img')
        self.extract_modify_tables(tables)
        self.extract_modify_images(images)
        html_path = self.HTML_HOME / f'{self.job_name}.html'
        with open(html_path, 'w') as file:
            file.write(str(self.soup))
        return self.file_outputs

    def get_webpage(self):
        response = requests.get(self.url)
        if "text/html" in response.headers["content-type"]:
            return response.text
        else:
            raise ValueError("URL does not contain HTML content")

    def extract_modify_images(self, images):
        for image_count, image in enumerate(images):
            image_url = image['src']
            img_data = requests.get(self.url+'/'+image_url).content
            file_path = self.IMG_HOME / f'{self.job_name}-img{image_count}.jpg'
            self.file_outputs['image_count'] += 1
            with open(file_path, 'wb') as handler:
                handler.write(img_data)
            # new_tag = self.soup.new_tag("p")
            # new_tag.string = f'![Image {image_count}]({file_path})'
            # image.replace_with(new_tag)

    def extract_modify_tables(self, tables: ResultSet):
        for table_count, table in enumerate(tables):
            read_table = pd.read_html(StringIO(str(table)))
            df = read_table[0]
            file_path = self.CSV_HOME / f'{self.job_name}-table{table_count}.csv'
            self.file_outputs['table_count'] += 1
            df.to_csv(file_path)

if __name__ == '__main__':
    modified_html = WebScraper('https://www.crummy.com/software/BeautifulSoup/bs4/doc/', 'bs4docs' ).extract_all()
