"""
Document ingestion pipeline

This file turns local files into structured chunk JSON.

High-level flow:

raw document -> read text -> clean text -> split into chunks -> save chunk JSON
"""

import json
import re
from pathlib import Path


RAW_DOCUMENTS_DIR = Path("/Users/howardzhao/personal_brain/data/raw_documents")
PROCESSED_CHUNKS_DIR = Path("/Users/howardzhao/personal_brain/data/processed_chunks")

SUPPORTED_EXTENSIONS = {".txt", ".md"}

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def list_raw_documents():
    """
    Find all supported raw documents in RAW_DOCUMENTS_DIR.

    Return a list of Path objects.
    """
    doc_path = RAW_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    doc_lst = []

    for file in RAW_DOCUMENTS_DIR.iterdir():
        if file.is_file() and file.suffix in SUPPORTED_EXTENSIONS:
            doc_lst.append(file)
    
    return doc_lst


def read_document(document_path):
    """
    Read one document from disk and return its text.
    """
    with open(document_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    return text

def clean_text(text):
    """
    Clean raw document text before chunking.
    """
    # normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # remove extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """
    Split cleaned text into smaller overlapping chunks.

    chunk_size means the approximate maximum number of characters per chunk.
    chunk_overlap means how many characters from the previous chunk are repeated.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    
    return chunks


def build_chunk_records(document_path, chunks):
    """
    Convert plain chunk strings into structured dictionaries.
    """
    records = []
    for index, chunk in enumerate(chunks):
        chunk_id = document_path.stem + "_" + f"{index:04d}"
        chunk_dict = {
            "chunk_id" : chunk_id,
            "chunk_index" : index,
            "text" : chunk
        }
        records.append(chunk_dict)
    
    return records


def build_output_path(document_path):
    """
    Helper Function

    Build the output JSON path for one source document.
    """
    PROCESSED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    output_file_name = document_path.stem + "_chunks.json"
    output_path = PROCESSED_CHUNKS_DIR / output_file_name
    
    return output_path


def save_chunks(document_path, chunk_records):
    """
    Save chunk records for one document as JSON.
    """
    output_path = build_output_path(document_path)
    data={
        "source_file" : document_path.name,
        "source_path" : str(document_path),
        "chunk_count" : len(chunk_records),
        "chunks" : chunk_records
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    
    return output_path


def ingest_document(document_path):
    """
    Run the full ingestion pipeline for one document.
    """
    text = read_document(document_path)
    text = clean_text(text)
    chunks = chunk_text(text)
    records = build_chunk_records(document_path, chunks)
    output_path = save_chunks(document_path, records)

    return output_path


def ingest_all_documents():
    """
    Ingest every supported document in RAW_DOCUMENTS_DIR.
    """
    output_paths = []

    doc_paths = list_raw_documents()

    for doc in doc_paths:
        expected_output_path = build_output_path(doc)
        if expected_output_path.exists():
            continue
        output_path = ingest_document(doc)
        output_paths.append(output_path) 
    
    return output_paths

if __name__ == "__main__":
    output_paths = ingest_all_documents()
    print(f"All files ingested and saved in {output_paths}.")
