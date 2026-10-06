import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# ===
DOCS_DIR = "docs"
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

print("📌 chargement et analyse des documents (Markdown et PDF)...")
documents = []

#===
for fileName in os.listdir(DOCS_DIR):
    filePath = os.path.join(DOCS_DIR,fileName)
    #==
    if fileName.endswith(".md"):
        with open(filePath,"r",encoding="utf-8") as f:
            textContent = f.read()
            if textContent.strip():
                documents.append(Document(pageContent = textContent,metadata = {"source":fileName}))
    #==
    elif fileName.endswith(".pdf"):
        try:
            loader = PyPDFLoader(filePath)
            pdfDocs = loader.load()
            documents.extend(pdfDocs)
            print(f"✅ Fichier PDF charge avec succes : {fileName}")
        except Exception as e:
            print(f"✖️ Erreur lors du chargement du PDF {fileName} : {e}")

print(f" Nombre total de fichiers trouves et charges en memoire : {len(documents)}")








