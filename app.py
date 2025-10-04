import pandas as pd
import os
import json
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import logging
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
from google.cloud import storage
import threading

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
# Read values from environment variables.
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
LLM_MODEL = "gemini-2.5-flash-preview-05-20"
EMBEDDING_MODEL_NAME = "text-embedding-004"

# These must be set as environment variables.
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
CSV_FILE_NAME = os.environ.get("CSV_FILE_NAME")
EMBEDDINGS_FILE_NAME = os.environ.get("EMBEDDINGS_FILE_NAME", "processed/embeddings.json")

# --- Globals and Initialization ---
app = Flask(__name__)
storage_client = storage.Client()

# Global variables to hold loaded data and state
INCIDENT_DATA = None
EMBEDDINGS_DATA = None
is_initialized = False # Flag to track successful initialization
initialization_lock = threading.Lock() # Lock for thread-safe initialization

# Initialize Vertex AI clients.
try:
    vertexai.init(project=PROJECT_ID, location=REGION)
    llm = GenerativeModel(LLM_MODEL)
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
except Exception as e:
    logging.error(f"Failed to initialize Vertex AI clients: {e}")
    llm = None
    embedding_model = None

# --- Data Loading and Processing Functions ---

def download_blob_to_tempfile(bucket_name, source_blob_name):
    """Downloads a blob from GCS to a temporary file."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    # Use tempfile to create a secure temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name
    temp_file.close()

    logging.info(f"Downloading {source_blob_name} to {temp_file_path}")
    blob.download_to_filename(temp_file_path)
    return temp_file_path

def load_data_from_gcs():
    """Downloads data from GCS and loads it into memory."""
    # Ensure all global variables are declared first to avoid SyntaxError
    global INCIDENT_DATA, EMBEDDINGS_DATA 
    
    if not GCS_BUCKET_NAME or not CSV_FILE_NAME or not EMBEDDINGS_FILE_NAME:
        logging.error("GCS environment variables are not set.")
        return False
        
    try:
        # 1. Load Incident Data (CSV)
        csv_path = download_blob_to_tempfile(GCS_BUCKET_NAME, CSV_FILE_NAME)
        df = pd.read_csv(csv_path)
        
        # --- Mandatory: Convert all context columns to string for RAG purposes ---
        # Ensure all columns needed for RAG context are treated as strings
        columns_to_convert = [
            'document_id', 'reporter_name', 'contact_type', 'category', 
            'item_affected', 'short_description', 'description', 
            'priority', 'status', 'assignment_group', 'sla_breached', 'root_cause'
        ]
        
        for col in columns_to_convert:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('')
            else:
                logging.warning(f"Column '{col}' not found in CSV data. Using empty string.")
                df[col] = '' # Add empty column if missing
        
        # Set 'document_id' as index for fast lookups
        INCIDENT_DATA = df.rename(columns={'document_id': 'Incident ID'}).set_index('Incident ID')
        
        logging.info(f"Loaded {len(INCIDENT_DATA)} incident records with enhanced metadata.")
        os.remove(csv_path) # Clean up temp file
        
        # 2. Load Embeddings Data (JSON)
        embeddings_path = download_blob_to_tempfile(GCS_BUCKET_NAME, EMBEDDINGS_FILE_NAME)
        with open(embeddings_path, 'r') as f:
            embedding_json = json.load(f)

        # Convert list of floats back to numpy arrays for fast vector operations
        EMBEDDINGS_DATA = {
            doc_id: np.array(data['embedding'])
            for doc_id, data in embedding_json.items()
        }
        logging.info(f"Loaded embeddings for {len(EMBEDDINGS_DATA)} documents.")
        os.remove(embeddings_path) # Clean up temp file
        
        return True
    
    except Exception as e:
        logging.error(f"Error loading data from GCS: {e}")
        return False

# This wrapper ensures the data loading only happens once, even if multiple
# containers start simultaneously due to Cloud Run scaling.
def initialize_global_resources():
    """A wrapper to load data and handle the main initialization logic."""
    # Ensure all global variables are declared first to avoid SyntaxError
    global EMBEDDINGS_DATA, INCIDENT_DATA, is_initialized

    if is_initialized:
        logging.info("Service already initialized.")
        return True

    # Use a lock to ensure only one thread performs initialization
    with initialization_lock:
        if is_initialized: # Check again after acquiring lock
            return True
            
        logging.info("Starting initial resource loading...")
        if load_data_from_gcs():
            if llm and embedding_model:
                is_initialized = True
                logging.info("Service initialization complete.")
                return True
            else:
                logging.error("Vertex AI models failed to initialize.")
                return False
        else:
            logging.error("Data loading failed.")
            return False

# --- RAG Core Functions ---

def get_query_embedding(query_text):
    """Generates an embedding for the user query."""
    try:
        if embedding_model is None:
            logging.error("Embedding model is not initialized.")
            return None
        
        response = embedding_model.get_embeddings([query_text])
        # The result is a list of embeddings; we take the first one (as the input was a single query)
        # and convert it to a NumPy array for compatibility with EMBEDDINGS_DATA
        return np.array(response[0].values)
    except Exception as e:
        logging.error(f"Error generating query embedding: {e}")
        return None

def retrieve_incidents_in_memory(query_embedding_values, top_k=5):
    """
    Performs similarity search using cosine similarity against the loaded embeddings.
    Returns the top_k most similar incident documents with full metadata.
    """
    if EMBEDDINGS_DATA is None or INCIDENT_DATA is None:
        return []

    # Prepare data for batch processing
    doc_ids = list(EMBEDDINGS_DATA.keys())
    embeddings = np.array(list(EMBEDDINGS_DATA.values()))
    
    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding_values.reshape(1, -1),
        embeddings
    )[0] 
    
    # Get the indices of the top_k most similar documents
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    retrieved_incidents = []
    
    for idx in top_indices:
        doc_id = doc_ids[idx]
        similarity_score = similarities[idx]
        
        if doc_id in INCIDENT_DATA.index:
            incident = INCIDENT_DATA.loc[doc_id]
            
            # Construct a detailed context string including ALL requested fields
            context_string = (
                f"Incident ID: {doc_id}\n"
                f"Similarity Score: {similarity_score:.4f}\n"
                f"Reporter Name: {incident['reporter_name']}\n"
                f"Contact Type: {incident['contact_type']}\n"
                f"Category: {incident['category']}\n"
                f"Item Affected (CI): {incident['item_affected']}\n"
                f"Priority: {incident['priority']}\n"
                f"Status: {incident['status']}\n"
                f"Assignment Group: {incident['assignment_group']}\n"
                f"SLA Breached: {incident['sla_breached']}\n"
                f"Root Cause: {incident['root_cause']}\n"
                f"Short Description: {incident['short_description']}\n"
                f"Full Description: {incident['description']}\n" # Assuming 'description' is the full text
            )
            
            # Also prepare a simplified dictionary for the frontend to display context
            context_dict = {
                "incident_id": doc_id,
                "short_description": incident['short_description'],
                "context_string": context_string
            }
            retrieved_incidents.append(context_dict)
            
    # The return structure is a list of dictionaries, where each dict contains 
    # the full context string for the LLM and the parts needed for the UI.
    return retrieved_incidents

def generate_rag_answer(user_query, retrieved_incidents):
    """Generates an answer using the LLM, grounded by the retrieved context."""
    if llm is None:
        return "The Language Model failed to initialize. Cannot generate a response."
        
    # Extract only the full context strings for the LLM prompt
    context_text = "\n\n---\n\n".join([item['context_string'] for item in retrieved_incidents])
    
    if not retrieved_incidents:
        # If no relevant incidents are found, inform the LLM and ask it to respond based on general knowledge
        prompt = (
            "You are an ITSM Executive Assistant. I was unable to retrieve any relevant "
            "internal incidents for the following user query. Based only on your general "
            "knowledge of IT Service Management (ITSM), provide a concise, professional, "
            "and helpful answer to the user query. If you cannot provide a helpful answer, "
            "politely state that the query requires information not available in your system."
            f"\n\nUser Query: {user_query}"
        )
    else:
        # Use retrieved context for grounding
        prompt = (
            "You are an ITSM Executive Assistant. Use the provided Incident Data below as "
            "your context to answer the user's query. "
            "Follow these rules:\n"
            "1. Base your answer *strictly* on the provided context.\n"
            "2. If the context does not contain enough information to answer, state that you "
            "cannot answer based on the available data.\n"
            "3. Provide a clear, professional, and concise answer in plain language. Do not "
            "include the Incident ID, Similarity Score, or the '--' separators in your final answer.\n"
            "4. Combine and synthesize information from multiple incidents if necessary to form a complete answer.\n\n"
            f"--- Incident Data ---\n{context_text}\n\n"
            f"--- User Query ---\n{user_query}"
        )
    
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return "An error occurred while generating the LLM response."

# --- Flask Routes ---

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_index(path):
    """Serves the index.html file for the root path."""
    if path == "":
        return send_from_directory('.', 'index.html')
    return send_from_directory('.', 'index.html')

@app.route('/rag', methods=['POST'])
def rag_endpoint():
    """Handles the RAG query requests."""
    
    # 1. Initialize resources on the first request
    if not initialize_global_resources():
        return jsonify({"error": "Service initialization failed. Could not load required data."}), 500
        
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body."}), 400
        
        user_query = data['query']
        
        # 2. Get the query embedding
        query_embedding_values = get_query_embedding(user_query)
        if query_embedding_values is None:
            return jsonify({"error": "Failed to generate query embedding."}), 500
            
        # 3. Retrieve context
        retrieved_incidents = retrieve_incidents_in_memory(query_embedding_values)
        
        # 4. Generate RAG answer
        rag_answer = generate_rag_answer(user_query, retrieved_incidents)
        
        # Check if the model's answer indicates a lack of information
        if "cannot answer" in rag_answer.lower() or "not sufficient" in rag_answer.lower():
            logging.info(f"RAG_NO_ANSWER_EVENT: Query='{user_query}'")
            
        # 5. Return the answer and the context for display
        return jsonify({
            "answer": rag_answer,
            # Send context to the frontend: only the dictionary elements are needed, 
            # as the full context string is only for the LLM.
            "context": [{"incident_id": item['incident_id'], "short_description": item['short_description']} for item in retrieved_incidents] 
        })
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during the RAG process: {e}")
        return jsonify({"error": f"An unexpected error occurred during the RAG process: {e}"}), 500

# --- Main Entry Point ---

# Check if running locally (not in Cloud Run), for easier local testing
if __name__ == '__main__':
    logging.info("Starting local initialization...")
    # Attempt to initialize resources immediately when running locally
    if initialize_global_resources():
        logging.info("Starting Flask server on port 8080...")
        # Note: Cloud Run uses gunicorn to start the app, so this block is mainly for local dev.
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    else:
        logging.error("Failed to initialize resources. Exiting.")
