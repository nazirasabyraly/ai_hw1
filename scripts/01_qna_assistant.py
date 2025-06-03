import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_assistant_id():
    with open("assistant_id.txt", "r") as f:
        return f.read().strip()

def create_thread():
    return client.threads.create()

def check_answer_has_chunk_id(answer):
    # Look for patterns like "chunk 123" or "section ABC" in the answer
    return any(marker in answer.lower() for marker in ["chunk", "section"])

def get_answer(thread, question):
    # Add the user's message to the thread
    message = client.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question
    )
    
    # Run the assistant
    run = client.threads.runs.create(
        thread_id=thread.id,
        assistant_id=load_assistant_id()
    )
    
    # Wait for the response
    while True:
        run_status = client.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run_status.status == 'completed':
            break
        elif run_status.status == 'failed':
            return "Sorry, I encountered an error while processing your question."
        time.sleep(1)
    
    # Get the assistant's response
    messages = client.threads.messages.list(thread_id=thread.id)
    latest_response = messages.data[0].content[0].text.value
    
    # Validate that the answer references at least one chunk ID
    if not check_answer_has_chunk_id(latest_response):
        # If no chunk ID is found, ask the assistant to revise with citations
        return get_answer(thread, 
            question + "\n\nPlease revise your answer to include specific chunk or section references from the source material.")
    
    return latest_response

def main():
    print("Welcome to the Study Q&A Assistant!")
    print("Type 'exit' to end the conversation.")
    
    # Create a new thread for this conversation
    thread = create_thread()
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == 'exit':
            break
            
        print("\nThinking...")
        answer = get_answer(thread, question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
