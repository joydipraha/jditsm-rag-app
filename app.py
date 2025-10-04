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

# Set up basic logging to capture INFO, WARNING, and ERROR messages
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
LLM_MODEL = "gemini-2.5-flash-preview-05-20"

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
CSV_FILE_NAME = os.environ.get("CSV_FILE_NAME")
EMBEDDINGS_FILE_NAME = "processed/embeddings.json"

# Columns to be retrieved from the incident data to form the context for the LLM
CONTEXT_COLUMNS = [
    'reporter_name', 'contact_type', 'category', 'item_affected_ci', 
    'short_description', 'priority', 'status', 'assignment_group', 
    'sla_breached', 'root_cause'
]

# Mapping keys to display names for cleaner output
CONTEXT_DISPLAY_NAMES = {
    'incident_id': 'Incident ID',
    'reporter_name': 'Reporter Name',
    'contact_type': 'Contact Type',
    'category': 'Category',
    'item_affected_ci': 'Item Affected (CI)',
    'short_description': 'Short Description',
    'priority': 'Priority',
    'status': 'Status',
    'assignment_group': 'Assignment Group',
    'sla_breached': 'SLA Breached',
    'root_cause': 'Root Cause'
}

# --- Globals and Initialization ---
app = Flask(__name__)
storage_client = None
INCIDENT_DATA = None
EMBEDDINGS_DATA = None

# Initialize Vertex AI clients.
try:
    vertexai.init(project=PROJECT_ID, location=REGION)
    llm = GenerativeModel(LLM_MODEL)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
except Exception as e:
    logging.error(f"Vertex AI Initialization Failed: {e}")


# --- Utility Functions for GCS ---

def download_and_load_csv(client, bucket_name, source_blob_name):
    """Downloads a CSV from GCS and loads it into a pandas DataFrame."""
    full_gcs_path = f"gs://{bucket_name}/{source_blob_name}"
    logging.info(f"Attempting to download CSV (data file): {full_gcs_path}")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    with tempfile.NamedTemporaryFile() as temp_file:
        blob.download_to_filename(temp_file.name)
        temp_file.seek(0)
        # Assuming the CSV contains the required columns
        df = pd.read_csv(temp_file.name)
    
    logging.info(f"Successfully loaded CSV with {len(df)} rows.")
    return df.set_index('incident_id')

def download_and_load_json(client, bucket_name, source_blob_name):
    """Downloads a JSON (embeddings) from GCS and loads it."""
    full_gcs_path = f"gs://{bucket_name}/{source_blob_name}"
    logging.info(f"Attempting to download JSON (embeddings file): {full_gcs_path}")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    json_data = blob.download_as_text()
    data = json.loads(json_data)

    logging.info(f"Successfully loaded JSON with {len(data)} items.")
    return data


# --- RAG Core Functions ---

def get_query_embedding(query_text: str) -> np.ndarray | None:
    """Generates an embedding for a text query."""
    try:
        response = embedding_model.get_embeddings(
            requests=[query_text]
        )
        return np.array(response[0].values)
    except Exception as e:
        logging.error(f"Error generating query embedding: {e}")
        return None

def retrieve_incidents_in_memory(query_embedding: np.ndarray, top_k: int = 5) -> list:
    """Performs in-memory similarity search and retrieves specific context columns."""
    if EMBEDDINGS_DATA is None or INCIDENT_DATA is None:
        return []
    
    embedding_matrix = np.array(EMBEDDINGS_DATA)
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embedding_matrix)[0]
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_incidents = []
    
    for idx in top_k_indices:
        incident_id = INCIDENT_DATA.index[idx]
        incident_row = INCIDENT_DATA.loc[incident_id].to_dict()
        
        # Build the context dictionary using required fields
        incident_context = {"incident_id": incident_id}
        for col in CONTEXT_COLUMNS:
            # Use .get() to avoid crashing if a column is missing in the CSV
            incident_context[col] = incident_row.get(col, "N/A")
            
        retrieved_incidents.append(incident_context)
        
    return retrieved_incidents

def generate_rag_answer(query: str, incidents: list) -> str:
    """Generates a response using Gemini based on the retrieved context."""
    if not incidents:
        return "I cannot answer your question as no relevant incidents were found in the knowledge base."
    
    # Format the context using all retrieved fields
    context_entries = []
    for inc in incidents:
        context_lines = [f"{CONTEXT_DISPLAY_NAMES[k]}: {v}" for k, v in inc.items()]
        context_entries.append("\n".join(context_lines))

    context_text = "\n\n---\n\n".join(context_entries)

    system_prompt = (
        "You are an ITSM Executive Assistant. Your goal is to answer the user's question "
        "based **ONLY** on the context provided below. Be concise and professional. "
        "If the context does not contain enough information to answer the question, "
        "politely state that you **cannot answer** based on the provided data."
    )

    prompt = (
        f"User Query: {query}\n\n"
        f"Context Incidents:\n{context_text}"
    )

    try:
        response = llm.generate_content(
            contents=prompt,
            config={"system_instruction": system_prompt}
        )
        return response.text
    except Exception as e:
        logging.error(f"Error generating RAG answer: {e}")
        return "An internal error occurred while generating the response."


# --- Initialization Logic ---

def initialize_global_resources():
    """Initializes the global DataFrame and Embeddings array."""
    global INCIDENT_DATA, EMBEDDINGS_DATA, storage_client
    
    if INCIDENT_DATA is not None and EMBEDDINGS_DATA is not None:
        logging.info("Resources already initialized.")
        return True
        
    if GCS_BUCKET_NAME is None or CSV_FILE_NAME is None:
        logging.critical("CRITICAL: GCS environment variables are not set.")
        return False
        
    try:
        if storage_client is None:
             storage_client = storage.Client()
        
        logging.info("--- Starting Resource Initialization ---")
        
        # 1. Download and load the main data frame (CSV)
        INCIDENT_DATA = download_and_load_csv(storage_client, GCS_BUCKET_NAME, CSV_FILE_NAME)
        
        # 2. Download and load the embeddings (JSON)
        embeddings_list = download_and_load_json(storage_client, GCS_BUCKET_NAME, EMBEDDINGS_FILE_NAME)
        
        # 3. Convert the list of lists into a single NumPy array
        global EMBEDDINGS_DATA
        EMBEDDINGS_DATA = np.array(embeddings_list)
        
        logging.info(f"Embeddings array shape: {EMBEDDINGS_DATA.shape}")
        
        # Final validation check
        if len(INCIDENT_DATA) != EMBEDDINGS_DATA.shape[0]:
            logging.critical(f"Data Mismatch: {len(INCIDENT_DATA)} rows in CSV vs {EMBEDDINGS_DATA.shape[0]} embeddings. RAG will not work correctly.")
            return False

        logging.info("--- Resource Initialization Complete ---")
        return True
        
    except Exception as e:
        # **This is where the GCS error is caught (likely 404)**
        logging.error(f"FATAL: Service initialization failed during GCS download/load: {type(e).__name__}: {e}")
        return False


# --- App Endpoints ---

@app.route('/', methods=['GET'])
def home():
    """Serves the front-end HTML for testing."""
    return send_from_directory('.', 'index.html')

@app.route('/rag', methods=['POST'])
def rag_endpoint():
    """Handles the RAG query."""
    if not initialize_global_resources():
        return jsonify({"error": "Service initialization failed. Could not load required data. Check Cloud Run logs for FATAL error details (e.g., GCS 404)."}), 500
        
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body."}), 400
        
        user_query = data['query']
        query_embedding_values = get_query_embedding(user_query)
        
        if query_embedding_values is None:
            return jsonify({"error": "Failed to generate query embedding."}), 500
            
        retrieved_incidents = retrieve_incidents_in_memory(query_embedding_values)
        rag_answer = generate_rag_answer(user_query, retrieved_incidents)
        
        # Format the context for the frontend using display names
        frontend_context = []
        for inc in retrieved_incidents:
            formatted_inc = {}
            for key, display_name in CONTEXT_DISPLAY_NAMES.items():
                formatted_inc[display_name] = inc.get(key, "N/A")
            frontend_context.append(formatted_inc)

        return jsonify({
            "answer": rag_answer, 
            "context": frontend_context
        })
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during the RAG process: {e}")
        return jsonify({"error": f"An unexpected error occurred during the RAG process: {e}"}), 500

# --- Main Entry Point ---

if __name__ == '__main__':
    if initialize_global_resources():
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
