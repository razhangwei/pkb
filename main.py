import logging
import os
import hashlib
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the path for storing the index
PERSIST_DIR = "./stored_index"

# Set up Ollama embedding model
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Configure global settings to use Ollama embeddings
Settings.embed_model = embed_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_file_hash(filepath):
    """Calculate and return the MD5 hash of a file"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_documents(directory):
    """Load documents from the specified directory"""
    logging.info(f"Loading documents from {directory}")
    documents = SimpleDirectoryReader(directory).load_data()
    logging.info(f"Loaded {len(documents)} documents")
    return documents

def create_index(documents, verbose=False):
    """Create an index from the loaded documents"""
    logging.info("Creating index from documents")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=verbose,
        embed_model=Settings.embed_model  # Use the globally configured embed model
    )
    logging.info("Index creation completed")
    return index

def save_index(index):
    """Save the index to disk"""
    logging.info(f"Saving index to {PERSIST_DIR}")
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    logging.info("Index saved successfully")

def load_index():
    """Load the index from disk"""
    logging.info(f"Loading index from {PERSIST_DIR}")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    logging.info("Index loaded successfully")
    return index

def update_index(index, doc_dir):
    """Update the index with new or modified documents"""
    logging.info("Updating index")
    documents = SimpleDirectoryReader(doc_dir).load_data()
    
    # Get existing document hashes
    existing_docs = {doc.metadata['file_path']: doc.metadata.get('file_hash') for doc in index.ref_doc_info.values()}
    
    # Update document metadata with file hash
    updated_docs = set()
    for doc in documents:
        file_path = doc.metadata['file_path']
        new_hash = get_file_hash(file_path)
        doc.metadata['file_hash'] = new_hash
        if file_path not in existing_docs or existing_docs[file_path] != new_hash:
            updated_docs.add(file_path)

    # Refresh the index with the new/updated documents
    refreshed_docs = index.refresh_ref_docs(
        documents,
        update_kwargs={"delete_kwargs": {"delete_from_docstore": True}}
    )

    # Log the refresh results
    for doc, was_refreshed in zip(documents, refreshed_docs):
        file_path = doc.metadata['file_path']
        if was_refreshed and file_path in updated_docs:
            logging.info(f"Updated/Inserted document: {file_path}")

    save_index(index)
    logging.info("Index update completed")

def query_index(index, query):
    """Query the index and return the response"""
    logging.info(f"Querying index with: {query}")
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    logging.info("Query completed")
    return response.response

def main():
    # Specify the directory containing your markdown notes
    notes_directory = "/Users/wei/Obsidian/Family/Parenting/Infant Care"

    # Check if the index already exists
    if os.path.exists(PERSIST_DIR):
        # Load the existing index
        index = load_index()
        # Update the index with any changes
        update_index(index, notes_directory)
    else:
        # Load documents and create a new index
        documents = load_documents(notes_directory)
        index = create_index(documents, verbose=True)
        # Save the new index
        save_index(index)

    # Example query
    query = "What are burping methods?"
    response = query_index(index, query)
    print(f"Query: {query}")
    print(f"Response: {response}")

    logging.info("Main execution completed")

if __name__ == "__main__":
    main()
