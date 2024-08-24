import logging
import os
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
