import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI

def load_documents(directory):
    """Load documents from the specified directory"""
    return SimpleDirectoryReader(directory).load_data()

def create_index(documents):
    """Create an index from the loaded documents"""
    return VectorStoreIndex.from_documents(documents)

def query_index(index, query):
    """Query the index and return the response"""
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response.response

def main():
    # Specify the directory containing your markdown notes
    notes_directory = "/Users/wei/Obsidian/Family"

    # Load documents
    documents = load_documents(notes_directory)

    # Create index
    index = create_index(documents)

    # Example query
    query = "What's my son's name?"
    response = query_index(index, query)
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
