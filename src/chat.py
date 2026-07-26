"""
Terminal chat interface for the Personal Brain assistant.

This file talks to the human.
It should not know about HTTP, JSON, llama-server endpoints, or model internals.
"""
import re
from llm_client import stream_llm


def print_welcome():
    """
    Print a startup message for the terminal app.
    """
    print("Hello, welcome to your personal brain! I am here to assist you. How can I help you today?")
    print("Tell the chatbot to exit to exit the program")


def is_exit_command(user_message):
    """
    Return True if the user wants to quit.
    Return False otherwise.
    """
    normalized_message = user_message.strip().lower()

    return re.search(r"\b(exit|quit|bye)\b", normalized_message) is not None


def handle_assistant_response(user_message):
    """
    Send the user message to the LLM and print streamed chunks.
    """
    print("Assistant: ", end="")
    response = stream_llm(user_message)
    for chunck in response:
        print(chunck, end="", flush=True)


def chat_loop():
    """
    Run the main terminal chat loop.
    """
    print_welcome()

    while True:
        user_input = input("Type your query here: ")
        if not user_input.strip():
            continue
        if is_exit_command(user_input) is True:
            break
        handle_assistant_response(user_input)
        print()

if __name__ == "__main__":
    chat_loop()
