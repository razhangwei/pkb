import logging
import os
import hashlib
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    QueryBundle,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv


# Load environment variables from .env file
def load_env():
    load_dotenv(override=True)


load_env()

# Get notes directories from environment variable
notes_directories_str = os.getenv("NOTES_DIRECTORIES", "")
notes_directories = [
    dir.strip() for dir in notes_directories_str.split(",") if dir.strip()
]

# Define the path for storing the index
PERSIST_DIR = "./stored_index"

# Set up Ollama embedding model
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Configure global settings to use Ollama embeddings
Settings.embed_model = embed_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_file_hash(filepath: str) -> str:
    """Calculate and return the MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_documents(directory: str, recursive: bool = True) -> List:
    """Load documents recursively from the specified directory and its subdirectories"""
    logging.info(f"Loading documents from {directory}")
    documents = SimpleDirectoryReader(directory, recursive=recursive).load_data()
    logging.info(f"Loaded {len(documents)} documents")
    return documents


def create_index(documents: List, verbose: bool = False) -> VectorStoreIndex:
    """Create an index from the loaded documents"""
    logging.info("Creating index from documents")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=verbose,
        embed_model=Settings.embed_model,  # Use the globally configured embed model
    )
    logging.info("Index creation completed")
    return index


def save_index(index: VectorStoreIndex) -> None:
    """Save the index to disk"""
    logging.info(f"Saving index to {PERSIST_DIR}")
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    logging.info("Index saved successfully")


def load_index() -> VectorStoreIndex:
    """Load the index from disk"""
    logging.info(f"Loading index from {PERSIST_DIR}")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    logging.info("Index loaded successfully")
    return index


def update_index(index: VectorStoreIndex, doc_dirs: List[str]) -> None:
    """Update the index with new or modified documents from multiple directories"""
    logging.info("Updating index")

    # Get existing document information
    existing_docs = {
        doc.metadata["file_path"]: doc.metadata.get("file_hash")
        for doc in index.docstore.docs.values()
    }

    # Load all documents from the directories
    all_documents = []
    for doc_dir in doc_dirs:
        all_documents.extend(SimpleDirectoryReader(doc_dir).load_data())

    # Identify new or modified documents
    documents_to_update = []
    for doc in all_documents:
        file_path = doc.metadata["file_path"]
        new_hash = get_file_hash(file_path)

        if file_path not in existing_docs or existing_docs[file_path] != new_hash:
            doc.metadata["file_hash"] = new_hash
            documents_to_update.append(doc)

    if not documents_to_update:
        logging.info("No documents need updating")
        return

    # Update the index with only the new/updated documents
    index.refresh_ref_docs(documents_to_update)

    save_index(index)
    logging.info(
        f"Index update completed. Updated {len(documents_to_update)} documents."
    )


def query_index(
    index: VectorStoreIndex,
    query: str,
    model_name: str,
    context: str = "",
    top_k: int = 4,
) -> str:
    """Query the index and return the response"""
    logging.info(f"Querying index with: {query} using model: {model_name}")

    # Initialize LLM
    if model_name.startswith("gemini"):
        llm = Gemini(model_name="models/" + model_name)
    else:
        llm = OpenAI(model=model_name)

    # Create a custom retriever with tunable top_k
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    # Create a query engine with the specified LLM and retriever
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever, llm=llm, verbose=True
    )

    # Prepare the full query with context
    full_query = f"Context:\n{context}\n\nQuery: {query}"

    # Query the index
    response = query_engine.query(QueryBundle(full_query))

    logging.info("Query completed")
    logging.info(f"Response: {response.response}")

    return response.response


def main(query: str, reindex: bool, model_name: str):
    # Use the notes_directories from the environment variable
    if not notes_directories:
        logging.error(
            "No notes directories specified in the NOTES_DIRECTORIES environment variable."
        )
        return

    # Check if the index already exists
    if os.path.exists(PERSIST_DIR):
        # Load the existing index
        index = load_index()
        if reindex:  # Change from update_index to reindex
            # Update the index with any changes
            update_index(index, notes_directories)
    else:
        # Load documents recursively and create a new index
        documents = []
        for directory in notes_directories:
            documents.extend(load_documents(directory))
        index = create_index(documents, verbose=True)
        # Save the new index
        save_index(index)

    # Query the index
    response = query_index(index, query, model_name)
    print(f"Query: {query}")
    print(f"Response: {response}")

    logging.info("Main execution completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the index and optionally update it."
    )
    parser.add_argument("query", type=str, help="The query to search for in the index")
    parser.add_argument(
        "--reindex", action="store_true", help="Update the index before querying"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="Specify the model to use for querying",
    )

    args = parser.parse_args()

    main(args.query, args.reindex, args.model)
