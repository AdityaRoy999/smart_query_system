import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

# Import the logic functions from your utils.py file
from utils import extract_text, chunk_text, embed_chunks, search_relevant_chunks, generate_response

# --- APP & SECURITY SETUP ---

# The 'app' object is what Vercel will look for and run
app = FastAPI(
    title="Smart Document Query API",
    description="An API to upload documents and ask questions using Gemini AI.",
    version="1.0.0"
)

# Define where to look for the API key (in the request header)
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Load the secret API keys from environment variables
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BACKEND_API_KEY or not GEMINI_API_KEY:
    # In a serverless environment, this might be handled differently,
    # but for now, it ensures keys are present.
    print("Warning: API keys are not set. The application might fail on Vercel if environment variables are not configured.")


# In-memory storage for processed documents.
# Note: This will reset with each serverless invocation on Vercel.
# For persistent storage, a database or a service like Vercel KV would be needed.
processed_files: Dict[str, list] = {}

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Dependency to validate the backend API key."""
    if api_key_header == BACKEND_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials",
        )

# --- REQUEST & RESPONSE MODELS ---

class ProcessResponse(BaseModel):
    file_id: str
    filename: str
    message: str
    total_chunks: int

class QueryRequest(BaseModel):
    file_id: str
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    relevant_chunks: list

# --- API ENDPOINTS ---

# NEW: Add a root endpoint for the base URL
@app.get("/hackerx/run")
def read_root():
    """A default endpoint for the root URL."""
    return {"message": "Smart Document Query API is active."}


@app.post("hackerx/run/process", response_model=ProcessResponse, dependencies=[Security(get_api_key)])
async def process_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF or DOCX), process it, and store its embeddings.
    """
    if not file.content_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or DOCX.")

    file_bytes = await file.read()
    file_id = f"{file.filename}-{len(file_bytes)}"

    if file_id in processed_files:
        return ProcessResponse(
            file_id=file_id,
            filename=file.filename,
            message="File has been previously processed.",
            total_chunks=len(processed_files[file_id])
        )

    try:
        full_text = extract_text(io.BytesIO(file_bytes), file.content_type)
        chunks = chunk_text(full_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract any text from the document.")

        # Pass the Gemini API key to the function
        embedded_chunks = embed_chunks(tuple(chunks), GEMINI_API_KEY)
        processed_files[file_id] = embedded_chunks

        return ProcessResponse(
            file_id=file_id,
            filename=file.filename,
            message="File processed successfully.",
            total_chunks=len(embedded_chunks)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")


@app.post("hackerx/run/query", response_model=QueryResponse, dependencies=[Security(get_api_key)])
async def query_document(request: QueryRequest):
    """
    Ask a question about a previously processed document.
    """
    if request.file_id not in processed_files:
        raise HTTPException(status_code=404, detail="File ID not found. Please process the document first.")

    try:
        embedded_chunks = processed_files[request.file_id]
        
        # Pass the Gemini API key to the functions
        top_chunks = search_relevant_chunks(request.query, embedded_chunks, GEMINI_API_KEY, top_k=request.top_k)
        if not top_chunks:
            raise HTTPException(status_code=404, detail="Could not find any relevant information for your query.")

        answer = generate_response(request.query, top_chunks, GEMINI_API_KEY)

        return QueryResponse(
            answer=answer,
            relevant_chunks=top_chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during query: {str(e)}")

# The uvicorn.run command is removed as Vercel handles the server execution.
