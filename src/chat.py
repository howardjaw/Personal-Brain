"""
Terminal chat interface for the Personal Brain assistant.

This file talks to the human.
It should not know about HTTP, JSON, llama-server endpoints, or model internals.
"""
import re
import json
from pathlib import Path
from llm_client import stream_llm, SYSTEM_PROMPT
from datetime import datetime

HISTORY_DIR = Path("/Users/howardzhao/personal_brain/data/conversation_history")


def print_welcome():
    """
    Print a startup message for the terminal app.
    """
    print("Hello, welcome to your personal brain! I am here to assist you. How can I help you today?")
    print("Tell the chatbot to exit to exit the program")

def create_session_path():
    """
    Create a conversation history path for the session
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M-%S")

    session_path = HISTORY_DIR / f"{timestamp}.json"

    return session_path

def create_initial_history():
    """
    Create initial history if there is no prior chat history.
    """
    initial_message=[]
    message = {
        "role" : "system",
        "content" : SYSTEM_PROMPT
    }
    initial_message.append(message)
    return initial_message

def load_history(session_path):
    """
    Load conversation history and read data
    """
    if not session_path.exists():
        return create_initial_history()
    with open(session_path, "r", encoding="utf-8") as file:
        history_messages = json.load(file)
    
    return history_messages

def save_history(session_path,messages):
    """
    Save the current message to the history.
    """
    with open(session_path, "w", encoding = "utf-8") as file:
        json.dump(messages, file, indent=2)


def is_exit_command(user_message):
    """
    Return True if the user wants to quit.
    Return False otherwise.
    """
    normalized_message = user_message.strip().lower()

    return re.search(r"\b(exit|quit|bye)\b", normalized_message) is not None


def handle_assistant_response(messages):
    """
    Send the user message to the LLM and print streamed chunks.
    """
    print("Assistant: ", end="")
    response = stream_llm(messages)
    assistant_chunks = []
    for chunck in response:
        print(chunck, end="", flush=True)
        assistant_chunks.append(chunck)
    print()
    assistant_answer = "".join(assistant_chunks)
    answer = {
        "role" : "assistant",
        "content" : assistant_answer
    }
    return answer
    

def build_stream_message(session_path, user_input):
    """
    Stream the user message to the client
    """
    messages = load_history(session_path)
    user_message = {
        "role" : "user",
        "content" : user_input
    }
    messages.append(user_message)
    return messages


def chat_loop():
    """
    Run the main terminal chat loop.
    """
    print_welcome()
    session_path = create_session_path()

    while True:
        user_input = input("Type your query here: ")
        if not user_input.strip():
            continue
        if is_exit_command(user_input) is True:
            break
        messages = build_stream_message(session_path, user_input)
        answer = handle_assistant_response(messages)
        messages.append(answer)
        save_history(session_path, messages)
        

if __name__ == "__main__":
    chat_loop()
