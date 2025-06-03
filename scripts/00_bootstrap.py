import os
import sys
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

def create_vector_store(client: OpenAI, store_name: str) -> dict:
    """Create a new vector store"""
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        print("✓ Vector store created:", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}

def upload_single_pdf(client: OpenAI, file_path: str, vector_store_id: str) -> dict:
    """Upload a single PDF file and attach it to the vector store"""
    file_name = file_path.name
    try:
        # Upload file
        file_response = client.files.create(
            file=open(file_path, 'rb'),
            purpose="assistants"
        )
        
        # Attach to vector store
        client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )
        
        return {
            "file": file_name,
            "file_id": file_response.id,
            "status": "success"
        }
    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")
        return {
            "file": file_name,
            "status": "failed",
            "error": str(e)
        }

def upload_pdf_files_to_vector_store(client: OpenAI, pdf_dir: Path, vector_store_id: str) -> dict:
    """Upload all PDF files from a directory to the vector store"""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    stats = {
        "total_files": len(pdf_files),
        "successful_uploads": 0,
        "failed_uploads": 0,
        "errors": [],
        "successful_file_ids": []
    }
    
    print(f"\nProcessing {len(pdf_files)} PDF files...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(upload_single_pdf, client, pdf_file, vector_store_id): pdf_file 
            for pdf_file in pdf_files
        }
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(pdf_files)):
            result = future.result()
            if result["status"] == "success":
                stats["successful_uploads"] += 1
                stats["successful_file_ids"].append(result["file_id"])
            else:
                stats["failed_uploads"] += 1
                stats["errors"].append(result)
    
    return stats

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
    
    # Verify data directory exists and contains PDFs
    if not data_dir.exists() or not any(data_dir.glob("*.pdf")):
        print(f"Error: No PDF files found in {data_dir}")
        sys.exit(1)

    print("1. Creating vector store...")
    vector_store = create_vector_store(client, "study_materials")
    if not vector_store:
        print("Failed to create vector store")
        sys.exit(1)

    print("\n2. Uploading and processing PDF files...")
    upload_stats = upload_pdf_files_to_vector_store(client, data_dir, vector_store["id"])
    
    if upload_stats["failed_uploads"] > 0:
        print(f"\nWarning: {upload_stats['failed_uploads']} files failed to upload")
        for error in upload_stats["errors"]:
            print(f"- {error['file']}: {error['error']}")
    
    if not upload_stats["successful_file_ids"]:
        print("Error: No files were successfully uploaded")
        sys.exit(1)

    print("\n3. Creating assistant...")
    try:
        assistant = client.beta.assistants.create(
            name="Study Q&A Assistant",
            instructions=(
                "You are a helpful tutor. Use the knowledge in the attached PDF files to answer questions. "
                "Always cite specific sections or page numbers from the PDFs when possible. "
                "When generating notes or summaries, ensure they are concise and focused on key concepts."
            ),
            model="gpt-4o",
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store["id"]]
                }
            }
        )
        print(f"✓ Assistant created with ID: {assistant.id}")
    except Exception as e:
        print(f"Error creating assistant: {e}")
        sys.exit(1)

    print("\n4. Saving assistant ID...")
    try:
        with open(script_dir / "assistant_id.txt", "w") as f:
            f.write(assistant.id)
        print("✓ Assistant ID saved to assistant_id.txt")
    except Exception as e:
        print(f"Error saving assistant ID: {e}")
        sys.exit(1)

    print(f"\n✅ Setup completed successfully!")
    print(f"- {upload_stats['successful_uploads']} files uploaded")
    print(f"- Vector store ID: {vector_store['id']}")
    print(f"- Assistant ID: {assistant.id}")

if __name__ == "__main__":
    main()
