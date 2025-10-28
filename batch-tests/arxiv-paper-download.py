import requests
import fitz  # PyMuPDF

pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"

# Download PDF
response = requests.get(pdf_url)
with open("paper.pdf", "wb") as f:
    f.write(response.content)

# Extract text
doc = fitz.open("paper.pdf")
text = ""
for page in doc:
    text += page.get_text()

print(text[:1000])  # print first 1000 characters
