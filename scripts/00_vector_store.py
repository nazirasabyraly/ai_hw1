from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    raise Exception(f"Failed to initialize OpenAI client: {e}")

# Create vector store
try:
    vector_store = client.vector_stores.create(name="My Study Materials")
    print("✅ Created vector store:", vector_store.id)
except Exception as e:
    raise Exception(f"Failed to create vector store: {e}")

# Upload and attach file
file_path = "../data/calculus_basics.pdf"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

try:
    with open(file_path, "rb") as f:
        file_batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=[f]
        )
    print("✅ Uploaded file to vector store.")
    print("VECTOR_STORE_ID =", vector_store.id)
except Exception as e:
    raise Exception(f"Failed to upload file: {e}")