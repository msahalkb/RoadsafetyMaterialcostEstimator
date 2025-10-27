from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
PDF_SOURCE_FOLDER=str(Path(__file__).parent / "irc_standards")
DB_PERSIST_FOLDER=str(Path(__file__).parent.parent / "chroma_db")

# Ensure the chroma_db folder exists
Path(DB_PERSIST_FOLDER).mkdir(parents=True, exist_ok=True)
#script
def main():
    #3.Load PDF documents from the specified folder
    print(f"Loading PDF documents from '{PDF_SOURCE_FOLDER}'...")
    loader = DirectoryLoader(PDF_SOURCE_FOLDER, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        print(f"No PDF documents found in '{PDF_SOURCE_FOLDER}'. Exiting.")
        return
    print(f"Loaded {len(documents)} documents.")
    #4.Split documents into larger chunks to reduce API calls
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Increased from 1000
        chunk_overlap=100,  # Reduced from 150
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    #5.Create embeddings using Google Generative AI
    print("Initializing local embedding model(will download on first  run)..")
    model_name="all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name,model_kwargs={'device':'cpu'})
    print("Local model loaded successfully.")
    print(f"Creating and persisting vector store at '{DB_PERSIST_FOLDER}'...")

    #6.Create and persist the Chroma vector store
    db = Chroma.from_documents(chunks,embeddings, persist_directory=DB_PERSIST_FOLDER)
    print("\n--- SUCCESS ---")
    print(f"Your Knowledge Base been created and saved to '{DB_PERSIST_FOLDER}' folder.")
    print("You can now run your main application script.")
if __name__ == "__main__":
    main()