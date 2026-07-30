"""
Embed processed document chunks.

This file connects two parts of the system:

processed chunk JSON -> embedding_client -> embedded chunk JSON

It should not know how to split raw documents, and it should not know how to
store vectors in a vector database yet.
"""

import json
from pathlib import Path

from embedding_client import EMBEDDING_MODEL_NAME, embed_text

EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"

PROCESSED_CHUNKS_DIR = Path("/Users/howardzhao/personal_brain/data/processed_chunks/")
EMBEDDED_CHUNKS_DIR = Path("/Users/howardzhao/personal_brain/data/embedded_chunks")

SUPPORTED_EXTENSIONS = {".json"}

def list_processed_chunk_files():
    """
    Find all processed chunk JSON files.

    Return a list of Path objects.
    """
    create_dir = PROCESSED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    processed_chunks_files_lst = []

    for file in PROCESSED_CHUNKS_DIR.iterdir():
        if file.is_file() and file.suffix in SUPPORTED_EXTENSIONS:
            processed_chunks_files_lst.append(file)

    return processed_chunks_files_lst


def load_chunk_file(chunk_file_path):
    """
    Load one processed chunk JSON file.

    Return the parsed Python dictionary.
    """
    with open(chunk_file_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    
    return chunks


def build_embedded_output_path(chunk_file_path):
    """
    A helper function that builds the output JSON path for embedded chunks.

    Example:
    lecture1_chunks.json -> lecture1_embeddings.json
    """
    EMBEDDED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    output_file_name = chunk_file_path.stem + "_embeddings.json"
    output_path = EMBEDDED_CHUNKS_DIR / output_file_name
    
    return output_path


def embed_chunk_records(chunk_file):
    """
    Add embeddings to every chunk in a processed chunk file.

    Return a new list of embedded chunk dictionaries.
    """
    embedded_chunks = []

    for chunk in chunk_file["chunks"]:
        embedding = embed_text(chunk["text"])
        embedding_dict = {
            "chunk_id" : chunk['chunk_id'],
            "chunk_index" : chunk['chunk_index'],
            "text" : chunk['text'],
            "embedding" : embedding
        }
        
        embedded_chunks.append(embedding_dict)
    
    return embedded_chunks


def build_embedded_chunk_file(chunk_file, embedded_chunks):
    """
    Build the top-level JSON object for one embedded chunk file.
    """
    embedded_chunk_file = {
        "source_file" : chunk_file["source_file"],
        "source_path" : chunk_file["source_path"],
        "embedding_model_name" : EMBEDDING_MODEL_NAME,
        "chunk_count" : chunk_file["chunk_count"],
        "chunks" : embedded_chunks

    }

    return embedded_chunk_file


def save_embedded_chunks(output_path, embedded_chunk_file):
    """
    Save embedded chunks as JSON.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(embedded_chunk_file, file, indent=2, ensure_ascii=False)
    
    return output_path


def embed_chunk_file(chunk_file_path):
    """
    Run the full embedding pipeline for one processed chunk JSON file.
    """
    chunked_file = load_chunk_file(chunk_file_path)
    embedded_chunks = embed_chunk_records(chunked_file)
    embedded_chunk_file = build_embedded_chunk_file(chunked_file, embedded_chunks)
    output_path = build_embedded_output_path(chunk_file_path)
    save_embedded_chunks(output_path, embedded_chunk_file)

    return output_path


def embed_all_chunk_files():
    """
    Embed every processed chunk JSON file.
    """
    processed_chunks_files_lst = list_processed_chunk_files()

    output_paths = []

    for file in processed_chunks_files_lst:
        output_path = embed_chunk_file(file)
        
    output_paths.append(output_path)

    return output_paths


if __name__ == "__main__":
    output = embed_all_chunk_files()
    print("Embeddings done")
