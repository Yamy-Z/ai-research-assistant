import requests
import os
from pathlib import Path

API_URL = "http://localhost:8000/api/v1"

def upload_documents():
    """Upload all test documents."""
    docs_dir = Path("data/test_docs")
    
    for file_path in docs_dir.glob("*.txt"):
        print(f"Uploading {file_path.name}...")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'text/plain')}
            response = requests.post(
                f"{API_URL}/documents/upload",
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Uploaded: {result['chunks_created']} chunks created")
            else:
                print(f"✗ Error: {response.text}")

if __name__ == "__main__":
    upload_documents()