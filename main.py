import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    index = VectorStoreIndex.from_documents(documents, show_progress=verbose)
    logging.info("Index creation completed")
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

    # Load documents
    documents = load_documents(notes_directory)

    # Create index with verbose mode on
    index = create_index(documents, verbose=True)

    # Example query
    query = "What are burping methods?"
    response = query_index(index, query)
    print(f"Query: {query}")
    print(f"Response: {response}")

    logging.info("Main execution completed")

if __name__ == "__main__":
    main()
