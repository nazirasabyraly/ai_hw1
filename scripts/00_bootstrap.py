import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)

    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    pdf_path = data_dir / "calculus_basics.pdf"

    # Verify PDF exists
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)

    print("1. Uploading PDF file...")
    try:
        file = client.files.create(
            file=open(pdf_path, "rb"),
            purpose="assistants"
        )
        print(f"✓ File uploaded with ID: {file.id}")
    except Exception as e:
        print(f"Error uploading file: {e}")
        sys.exit(1)

    print("\n2. Creating assistant...")
    try:
        assistant = client.beta.assistants.create(
            name="Study Q&A Assistant",
            instructions=(
                "You are a helpful tutor. Use the knowledge in the attached PDF to answer questions. "
                "Always cite specific sections or page numbers from the PDF when possible. "
                "When generating notes or summaries, ensure they are concise and focused on key concepts."
            ),
            model="gpt-4o",
            tools=[{"type": "file_search"}],
            file_ids=[file.id] 
        )
        print(f"✓ Assistant created with ID: {assistant.id}")
    except Exception as e:
        print(f"Error creating assistant: {e}")
        sys.exit(1)

    print("\n3. Attaching file to assistant...")
    

    print("\n✅ Setup completed successfully!")

if __name__ == "__main__":
    main()
