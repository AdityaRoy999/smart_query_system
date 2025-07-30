import re
import io
import time
import numpy as np
import docx
import requests
from PyPDF2 import PdfReader

# --- DOCUMENT PARSING ---
def extract_text(file_path, file_type):
    """Extracts text from PDF or DOCX files."""
    if file_type == "application/pdf":
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {e}")
        return text
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise ValueError(f"Error reading DOCX file: {e}")
    else:
        raise ValueError("Unsupported file type.")

# --- TEXT CHUNKING ---
def chunk_text(text, max_tokens=200):
    """Splits text into chunks of a maximum token size."""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

# --- API CALLS ---
def get_embedding(text_chunk, api_key: str):
    """Generates embedding for a single text chunk."""
    embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = {"model": "models/embedding-001", "content": {"parts": [{"text": text_chunk}]}}
    
    max_retries = 3
    delay = 1
    for attempt in range(max_retries):
        try:
            resp = requests.post(embed_url, headers=headers, json=body, timeout=60)
            resp.raise_for_status()
            embedding = resp.json().get("embedding", {}).get("values")
            if not embedding:
                raise ValueError("API response did not contain embedding values.")
            return embedding
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError(f"API request failed for single embedding: {e}")
    return None

def embed_chunks(chunks, api_key: str):
    """Embeds a list of text chunks in batches."""
    batch_embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:batchEmbedContents?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    all_embedded_chunks = []
    BATCH_SIZE = 100 

    for i in range(0, len(chunks), BATCH_SIZE):
        chunk_batch = chunks[i:i + BATCH_SIZE]
        
        requests_list = [{"model": "models/embedding-001", "content": {"parts": [{"text": chunk}]}} for chunk in chunk_batch]
        body = {"requests": requests_list}

        max_retries = 3
        delay = 2
        for attempt in range(max_retries):
            try:
                resp = requests.post(batch_embed_url, headers=headers, json=body, timeout=180)
                resp.raise_for_status()
                
                embeddings = resp.json().get("embeddings", [])
                if len(embeddings) != len(chunk_batch):
                    raise ValueError("API Error: Mismatch in batch size.")

                batch_embedded = [
                    {"id": f"chunk_{i+j}", "text": chunk, "embedding": emb.get("values")}
                    for j, (chunk, emb) in enumerate(zip(chunk_batch, embeddings))
                    if emb.get("values")
                ]
                all_embedded_chunks.extend(batch_embedded)
                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise RuntimeError(f"Batch embedding failed: {e}")
    
    return all_embedded_chunks

def generate_response(query, relevant_chunks, api_key: str):
    """Generates a natural language response."""
    generation_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    context = "\n\n".join([f"Clause: {c['chunk']}" for c in relevant_chunks])
    
    prompt = f"""
    You are a professional assistant. Your task is to give a direct answer to a user's query based *only* on the relevant clauses provided.
    **User Query:** "{query}"
    **Relevant Clauses:** --- {context} ---
    **Instructions:** Start your answer directly with "Yes," "No," or state that the information isn't available. Do not use preambles like "According to the text...".
    """
    
    headers = {"Content-Type": "application/json"}
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        resp = requests.post(generation_url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Could not generate response: {e}"

# --- SEMANTIC SEARCH ---
def cosine_similarity(v1, v2):
    """Calculates cosine similarity."""
    v1, v2 = np.array(v1), np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def search_relevant_chunks(query, embedded_chunks, api_key: str, top_k=3):
    """Finds the most relevant chunks to a query."""
    if not embedded_chunks:
        return []
    query_embedding = get_embedding(query, api_key)
    scored_chunks = [{"chunk": chunk["text"], "score": cosine_similarity(query_embedding, chunk["embedding"])} for chunk in embedded_chunks]
    
    return sorted(scored_chunks, key=lambda x: x["score"], reverse=True)[:top_k]
