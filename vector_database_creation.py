# importing necessary libraries
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


## ---- Configuration ---- ##
DATA_FILE = "Data"
DATA_FILE = os.path.join(DATA_FILE, "HR_Policy.txt")
VECTORSTORE_DIR = "hr_policy_vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# Loading and Chunking the Data
def load_and_chunk_documents():
    """"Loads the HR policy data from the TXT file and splits it into smaller chunks."""
    print("\t --- Step 1: Loading and Chunking Data ---")

    if not os.path.exists(DATA_FILE):
        print(f"Data file {DATA_FILE} does not exist.")
        return None

    # Read the text file
    with open(DATA_FILE, 'r', encoding='utf-8') as file:
        full_text = file.read()
    
    # Split the text into chunks
    documents = [full_text]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= CHUNK_SIZE,
        chunk_overlap= CHUNK_OVERLAP
    )

    chunks = text_splitter.create_documents(documents)
    print(f"Data loaded and split into {len(chunks)} chunks.")
    return chunks

# Creating and embedding the Vector Store
def create_and_store_embeddings(chunks):
    """Creates a vector store from the chunks of data."""
    print("\t --- Step 2: Creating Vector Store ---")

    if not chunks:
        print("No chunks to create vector store.")
        return None

    # Initialize the embedding model from HuggingFace
    print(f"Loading embedding model: '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Create the Chroma vector store
    # This will process all chunks and store their vector representations.
    # It will be persisted to disk in the VECTORSTORE_DIR.
    print(f"Creating vector store in '{VECTORSTORE_DIR}'...")
    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    print("Vector store created and persisted successfully.")

# Manually download the dataset first as per the instructions in the function.
doc_chunks = load_and_chunk_documents()
if doc_chunks:
    create_and_store_embeddings(doc_chunks)
    
print("\n\t--- Data Ingestion Complete ---")
print(f"Vector store is ready in the '{VECTORSTORE_DIR}' directory.")