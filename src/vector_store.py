import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from embedding_client import embed_text


QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "personal_brain_chunks"

EMBEDDED_CHUNKS_DIR = Path("/Users/howardzhao/personal_brain/data/embedded_chunks")

SUPPORTED_EXTENSIONS = {".json"}


def create_client():
    """
    Create a Python client object that can talk to the local Qdrant server.
    """
    client = QdrantClient(url=QDRANT_URL)
    return client


def list_embedded_chunk_files():
    """
    Find all embedded chunk JSON files.

    Return a list of Path objects.
    """
    if not EMBEDDED_CHUNKS_DIR.exists():
        raise FileNotFoundError(f"Embedded chunks directory did not exist at: {EMBEDDED_CHUNKS_DIR}.")
    
    embedded_chunks_file_lst = []

    for file in EMBEDDED_CHUNKS_DIR.iterdir():
        if file.is_file() and file.suffix == ".json":
            embedded_chunks_file_lst.append(file)
    
    return embedded_chunks_file_lst


def load_embedded_chunk_file(chunk_file_path):
    """
    Load one embedded chunk JSON file.

    Return the parsed Python dictionary.
    """

    with open(chunk_file_path, "r", encoding="utf-8") as file:
        embedded_chunk_file = json.load(file)
    
    return embedded_chunk_file


def get_vector_size(embedded_chunk_file):
    """
    Find how many numbers are inside one embedding vector.

    Qdrant needs this number before it can create a collection.
    """
    first_chunk = embedded_chunk_file["chunks"][0]
    embedding = first_chunk["embedding"]
    vector_size = len(embedding)

    return vector_size

def ensure_collection(client, vector_size):
    """
    Make sure the Qdrant collection exists.

    If it does not exist yet, create it.
    If it already exists, leave it alone.
    """
    collections = client.get_collections().collections

    collection_names= []

    for collection in collections:
        collection_names.append(collection.name)
    
    if COLLECTION_NAME in collection_names:
        return
    
    client.create_collection(
        collection_name = COLLECTION_NAME,
        vectors_config = VectorParams(
            size = vector_size,
            distance = Distance.COSINE
        )
    )


def build_point_id(chunk):
    """
    Build a stable Qdrant point id for one chunk.

    Qdrant accepts UUID strings as point ids.
    The same chunk_id should always produce the same UUID.
    """
    chunk_id = chunk["chunk_id"]
    point_id = uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)
    
    return str(point_id)


def build_payload(embedded_chunk_file, chunk):
    """
    Build the JSON metadata that will be stored beside the vector.

    The vector database searches vectors, but the payload is what we show back
    to the user after search.
    """
    # TODO: create a dictionary containing:
    #       - source_file from embedded_chunk_file
    #       - source_path from embedded_chunk_file
    #       - embedding_model_name from embedded_chunk_file
    #       - chunk_id from chunk
    #       - chunk_index from chunk
    #       - text from chunk
    payload = {
        "source_file" : embedded_chunk_file["source_file"],
        "source_path" : embedded_chunk_file["source_path"],
        "embedding_model_name" : embedded_chunk_file["embedding_model_name"],
        "chunk_id" : chunk["chunk_id"],
        "chunk_index" : chunk["chunk_index"],
        "text" : chunk["text"]
    }

    return payload


def build_point(embedded_chunk_file, chunk):
    """
    Convert one embedded chunk dictionary into one Qdrant PointStruct.
    """
    point_id = build_point_id(chunk)
    vector = chunk["embedding"]
    payload = build_payload(embedded_chunk_file, chunk)
    
    point = PointStruct(
        id=point_id,
        vector=vector,
        payload=payload
    )
    
    return point


def build_points_from_file(embedded_chunk_file):
    """
    Convert all chunks from one embedded chunk file into Qdrant points.
    """
    points = []

    for chunk in embedded_chunk_file["chunks"]:
        point = build_point(embedded_chunk_file, chunk)
        points.append(point)
    
    return points


def upsert_embedded_chunk_file(client, chunk_file_path):
    """
    Store all chunks from one embedded chunk JSON file into Qdrant.

    Upsert means:
    - insert the point if it does not exist
    - update the point if it already exists
    """
    embedded_chunk_file = load_embedded_chunk_file(chunk_file_path)
    
    points = build_points_from_file(embedded_chunk_file)
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    return len(points)


def upsert_all_embedded_chunks():
    """
    Store every embedded chunk JSON file into Qdrant.
    """
    count = 0
    client = create_client()
    embedded_chunk_file_lst = list_embedded_chunk_files()
    first_file_path = embedded_chunk_file_lst[0]
    first_file_data = load_embedded_chunk_file(first_file_path)
    vector_size = get_vector_size(first_file_data)
    ensure_collection(client, vector_size)

    for file_path in embedded_chunk_file_lst:
        count += upsert_embedded_chunk_file(client, file_path)
    
    return count
    


def search_chunks(query_text, limit=5):
    """
    Search Qdrant for chunks similar to the user's query.

    This is the beginning of retrieval.

    query text -> query embedding -> vector search -> matching chunk payloads
    """
    client = create_client()
    query_vector = embed_text(query_text)
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )

    return results.points


if __name__ == "__main__":
    count = upsert_all_embedded_chunks()
    print(f"{count} chunks stored in Qdrant DB")
    query_text = input("Type in query: ")
    results = search_chunks(query_text, limit=5)

    for point in results:
        print(point.score)
        print(point.payload["source_file"])
        print(point.payload["text"])
