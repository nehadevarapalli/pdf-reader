import requests
import os
import csv
from dotenv import load_dotenv

load_dotenv()

pswd=os.getenv('fire_api') 

url = "https://api.firecrawl.dev/v1/scrape"

payload = {
    "url": "https://webscraper.io/test-sites", #update the URL here
    "actions": [],
    "formats": ["markdown"]
}
headers = {
    "Authorization": pswd,
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

#print(response.text)

data = response.json()

# Ensure output directories exist .Change for cloud
os.makedirs("out/text", exist_ok=True)
os.makedirs("out/img", exist_ok=True)
os.makedirs("out/csv", exist_ok=True)

# Save Markdown Content
if data.get("success") and "markdown" in data["data"]:
    markdown_text = data["data"]["markdown"]
    md_file_path = "out/text/text.md"
    with open(md_file_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_text)
    print(f"Markdown saved to {md_file_path}")

# Download and Save Images
if data.get("success") and "markdown" in data["data"]:
    import re

    # Extract image URLs using regex
    image_urls = re.findall(r'!\[.*?\]\((https?://[^\s]+)\)', markdown_text)

    for img_url in image_urls:
        img_name = os.path.basename(img_url).split("?")[0]  # Get image filename
        img_path = os.path.join("out/img", img_name)

        # Download and save the image
        img_response = requests.get(img_url, stream=True)
        if img_response.status_code == 200:
            with open(img_path, "wb") as img_file:
                for chunk in img_response.iter_content(1024):
                    img_file.write(chunk)
            print(f"Image saved: {img_path}")

# Extract and Save CSV Data (if available)
if "tables" in data["data"]:  
    for i, table in enumerate(data["data"]["tables"]):
        csv_file_path = f"out/csv/table_{i}.csv"
        with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(table["headers"])
            writer.writerows(table["rows"])
        print(f"CSV saved: {csv_file_path}")