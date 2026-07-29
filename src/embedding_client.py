"""
Client for talking to the local embedding server.

This file has one job:

text -> embedding server -> embedding vector

It should not know about raw documents, chunk JSON files, vector databases,
or chat sessions.
"""

import requests

EMBEDDING_SERVER_URL = "http://127.0.0.1:8081/v1/embeddings"
EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"
TIMEOUT_SECONDS = 120


def build_embedding_payload(text):
    """
    Convert plain text into the JSON payload expected by the embedding server.
    """
    payload = {
        "model" : EMBEDDING_MODEL_NAME,
        "input" : text,
    }

    return payload


def send_embedding_request(payload):
    """
    Send the embedding payload to the embedding server.

    Return the parsed JSON response.
    """
    response = requests.post(
        EMBEDDING_SERVER_URL,
        json=payload,
        timeout = TIMEOUT_SECONDS
    )

    response.raise_for_status()

    return response.json()


def parse_embedding_response(response_json):
    """
    Extract the embedding vector from the embedding server response.
    """
    data = response_json["data"]
    first_item = data[0]
    embeddings = first_item["embedding"]

    return embeddings



def embed_text(text):
    """
    Public function for embedding one piece of text.
    """
    payload = build_embedding_payload(text)
    response_json = send_embedding_request(payload)
    embeddings = parse_embedding_response(response_json)
    return embeddings


if __name__ == "__main__":
    text = "Overfitting happens when a model memorizes training data."
    embedding = embed_text(text)

    print("Embedding dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])
