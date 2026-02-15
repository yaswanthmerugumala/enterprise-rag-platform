import fitz
import os

def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            text = load_pdf(os.path.join(folder_path, file))
            documents.append({
                "text": text,
                "source": file
            })

    return documents
