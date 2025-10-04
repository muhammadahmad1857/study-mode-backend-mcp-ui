
import os
import glob
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

def build_vector_store(input_dir: str = "knowledge-base/", gemini_api_key: str = None):
    """
    Build a vector store from text documents in the specified input directory.

    Args:
        input_dir (str): Path to the directory containing documents.
        gemini_api_key (str): Google Gemini API key for embeddings.

    Returns:
        None
    """
    if gemini_api_key is None:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set in environment variables.")

    # Load documents
    documents = []
    folders = glob.glob(input_dir)
    for folder in folders:
        docs_loader = DirectoryLoader(
            folder,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        docs_folder = docs_loader.load()
        for doc in docs_folder:
            file_name = os.path.basename(doc.metadata.get("source", ""))
            doc.metadata["page_title"] = os.path.splitext(file_name)[0]
            documents.append(doc)

    print(f"[INFO] Loaded {len(documents)} documents")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key
    )

    ds_name = os.path.join("..", "shared_data", "vector_store")

    # Create the shared_data directory if it doesn't exist
    os.makedirs(os.path.dirname(ds_name), exist_ok=True)

    # Remove old vector store if it exists
    if os.path.exists(ds_name):
        import shutil
        shutil.rmtree(ds_name)
        print(f"[INFO] Removed old vector store at {ds_name}")

    # Create new vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=ds_name,
        client_settings=Settings(anonymized_telemetry=False)
    )
    print(f"Vector store created with {vector_store._collection.count()} chunks")
