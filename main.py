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
)

from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

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


def load_documents(directory: str) -> List:
    """Load documents from the specified directory"""
    logging.info(f"Loading documents from {directory}")
    documents = SimpleDirectoryReader(directory).load_data()
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
    existing_docs: Dict[str, Tuple[str, object]] = {
        doc.metadata["file_path"]: (doc.metadata.get("file_hash"), doc)
        for doc in index.ref_doc_info.values()
    }

    # Load all documents from the directories
    all_documents: List = []
    for doc_dir in doc_dirs:
        all_documents.extend(
            SimpleDirectoryReader(doc_dir).load_data(show_progress=True, num_workers=4)
        )

    # Identify new or modified documents
    documents_to_update: Dict[str, object] = {}
    for doc in all_documents:
        file_path = doc.metadata["file_path"]
        new_hash = get_file_hash(file_path)

        if file_path not in existing_docs or existing_docs[file_path][0] != new_hash:
            doc.metadata["file_hash"] = new_hash
            documents_to_update[file_path] = doc

    if not documents_to_update:
        logging.info("No documents need updating")
        return

    # Refresh the index with only the new/updated documents
    refreshed_docs = index.refresh_ref_docs(
        list(documents_to_update.values()),
        update_kwargs={"delete_kwargs": {"delete_from_docstore": True}},
    )

    # Log the refresh results
    updated_count = 0
    for doc, was_refreshed in zip(documents_to_update.values(), refreshed_docs):
        if was_refreshed:
            logging.info(f"Updated/Inserted document: {doc.metadata['file_path']}")
            updated_count += 1

    save_index(index)
    logging.info(f"Index update completed. Updated {updated_count} documents.")


def query_index(
    index: VectorStoreIndex,
    query: str,
    model_name: str,
) -> str:
    """Query the index and return the response"""
    logging.info(f"Querying index with: {query} using model: {model_name}")
    if model_name.startswith("gemini"):
        llm = Gemini(model_name="models/" + model_name)
    else:
        llm = OpenAI(model=model_name)
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(query)
    logging.info("Query completed")
    return response.response


def main(query: str, update_index: bool, model_name: str):
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
        if update_index:
            # Update the index with any changes
            update_index(index, notes_directories)
    else:
        # Load documents and create a new index
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
        "--update_index", action="store_true", help="Update the index before querying"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="Specify the model to use for querying",
    )

    args = parser.parse_args()

    main(args.query, args.update_index, args.model)
