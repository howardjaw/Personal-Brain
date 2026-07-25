# src/llm_client.py

"""
Client for talking to the local llama.cpp server.

This file should not know about documents, embeddings, vector databases,
or the future UI. Its only job is:

Python -> llama-server -> assistant answer
"""

import requests


LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "local-qwen"
TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = """
You are a helpful local personal knowledge assistant.
"""


def build_messages(user_message):
    """
    Convert a plain user message into chat messages.

    Return a list of message dictionaries.
    """
    messages = []

    system_message = {
        "role" : "system",
        "content" : SYSTEM_PROMPT
    }

    parsed_user_message = {
        "role" : "user",
        "content" : user_message
    }

    messages.append(system_message)
    messages.append(parsed_user_message)

    return messages


def build_payload(messages):
    """
    Convert chat messages into the JSON payload expected by llama-server.
    """
    payload = {
        "model" : MODEL_NAME,
        "messages" : messages,
        "temperature" : 0.7,
        "max_tokens" : 350,
        "stream" : False
    }

    return payload


def send_request(payload):
    """
    Send the payload to llama-server using HTTP POST.

    Return the parsed JSON response.
    """
    response = requests.post(
        LLAMA_SERVER_URL,
        json=payload,
        timeout = TIMEOUT_SECONDS
    )

    response.raise_for_status()

    return response.json()

def parse_response(response_json):
    """
    Extract the assistant's final answer from llama-server's JSON response.
    """
    choices = response_json["choices"]
    first_choice = choices[0]
    message = first_choice["message"]
    content = message["content"]

    return content


def ask_llm(user_message):
    """
    Main public function.

    Other parts of the project should call this function instead of calling
    llama-server directly.
    """
    messages = build_messages(user_message)
    payload = build_payload(messages)
    response_json = send_request(payload)
    answer = parse_response(response_json)

    return answer


if __name__ == "__main__":
    question = input("type your question here: ")
    answer = ask_llm(question)
    print(answer)
