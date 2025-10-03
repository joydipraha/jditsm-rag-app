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

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
# Read values from environment variables.
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
LLM_MODEL = "gemini-2.5-flash-preview-05-20"

# These must be set as environment variables.
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
CSV_FILE_NAME = os.environ.get("CSV_FILE_NAME")
EMBEDDINGS_FILE_NAME = "processed/embeddings.json"

# --- Globals and Initialization ---
app = Flask(__name__)
storage_client = storage.Client()
INCIDENT_DATA = None
EMBEDDINGS_DATA = None

# Initialize Vertex AI clients.
try:
    vertexai.init(project=PROJECT_ID, location=REGION)
    llm = GenerativeModel(LLM_MODEL)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
except Exception as e:
    logging.error(f"Vertex AI initialization failed: {e}")

# --- Helper Functions ---

def download_gcs_file(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from GCS."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        # Check if the blob exists before attempting to download
        if not blob.exists():
            logging.error(f"GCS Blob not found: gs://{bucket_name}/{source_blob_name}")
            return False

        blob.download_to_filename(destination_file_name)
        logging.info(f"File {source_blob_name} downloaded to {destination_file_name}.")
        return True
    except Exception as e:
        logging.error(f"Failed to download {source_blob_name} from GCS: {e}")
        return False

def load_data():
    """Loads CSV and Embeddings data from local temporary files."""
    global INCIDENT_DATA, EMBEDDINGS_DATA
    
    # Load Incident Data (CSV)
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_csv:
            if not download_gcs_file(GCS_BUCKET_NAME, CSV_FILE_NAME, tmp_csv.name):
                return False
            INCIDENT_DATA = pd.read_csv(tmp_csv.name)
            os.unlink(tmp_csv.name)
        logging.info("Incident data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading incident data: {e}")
        return False
        
    # Load Embeddings Data (JSON)
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_json:
            if not download_gcs_file(GCS_BUCKET_NAME, EMBEDDINGS_FILE_NAME, tmp_json.name):
                return False
            with open(tmp_json.name, 'r') as f:
                embeddings_json = json.load(f)
            
            # Convert list of lists back to list of NumPy arrays for efficient computation
            EMBEDDINGS_DATA = {
                'incident_id': embeddings_json['incident_id'],
                'embeddings': [np.array(e) for e in embeddings_json['embeddings']]
            }
            os.unlink(tmp_json.name)
        logging.info("Embeddings data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading embeddings data: {e}")
        return False
        
    return True

def initialize_global_resources():
    """Checks if data is loaded, loads it if not."""
    global INCIDENT_DATA, EMBEDDINGS_DATA
    if INCIDENT_DATA is None or EMBEDDINGS_DATA is None:
        logging.info("Data not loaded yet. Starting initial load...")
        return load_data()
    return True

def get_query_embedding(query: str):
    """Generates an embedding for the user query."""
    try:
        response = embedding_model.get_embeddings([query])
        return response[0].values
    except Exception as e:
        logging.error(f"Failed to get embedding for query: {e}")
        return None

def retrieve_incidents_in_memory(query_embedding_values, k=5):
    """
    Retrieves the top k most relevant incidents based on cosine similarity
    between the query embedding and incident embeddings, including all requested fields.
    """
    if INCIDENT_DATA is None or EMBEDDINGS_DATA is None:
        logging.error("Data not initialized for retrieval.")
        return []

    query_embedding = np.array(query_embedding_values).reshape(1, -1)
    
    # Convert list of embedding lists to NumPy array for efficient calculation
    incident_embeddings_array = np.array(EMBEDDINGS_DATA['embeddings'])

    # Calculate cosine similarity between query and all incident embeddings
    similarities = cosine_similarity(query_embedding, incident_embeddings_array)[0]

    # Get the indices of the top k most similar incidents
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # Define the columns to be included in the context for the LLM
    context_columns = [
        "Incident ID",
        "Reporter Name",
        "Contact Type",
        "Category",
        "Item Affected (CI)",
        "Short Description",
        "Priority",
        "Status",
        "Assignment Group",
        "SLA Breached",
        "Root Cause"
    ]

    retrieved_incidents = []
    for idx in top_k_indices:
        incident_id = EMBEDDINGS_DATA['incident_id'][idx]
        # Get the row from the main DataFrame
        incident_row = INCIDENT_DATA[INCIDENT_DATA['Incident ID'] == incident_id].iloc[0]
        
        # Select the key columns for context, using "N/A" if a column is missing
        context = {col: incident_row.get(col, "N/A") for col in context_columns}
        retrieved_incidents.append(context)
        
    return retrieved_incidents

def generate_rag_answer(user_query: str, incidents: list):
    """Generates an answer using Gemini, grounded by retrieved incident data."""
    context_text = "\n".join([str(inc) for inc in incidents])
    
    system_prompt = (
        "You are an expert ITSM (IT Service Management) analyst. Your task is to analyze the provided "
        "incident data (the 'Context') and generate a concise, professional, and actionable answer "
        "to the user's query. The context includes fields like Incident ID, Reporter Name, Category, Priority, "
        "Status, and Root Cause. Use this detailed data to justify your answer. "
        "If the context does not contain sufficient information to answer the question, "
        "state clearly that you cannot answer based on the provided data."
    )

    full_prompt = (
        f"User Query: {user_query}\n\n"
        f"Context (Retrieved Incident Data):\n---\n{context_text}\n---\n\n"
        "Analyze the context data and provide your professional answer."
    )

    try:
        response = llm.generate_content(
            contents=full_prompt,
            system_instruction=system_prompt
        )
        return response.text
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}")
        return f"Failed to generate RAG answer due to an API error: {e}"

# --- Routes ---

@app.route('/')
def serve_index():
    """
    Serves the main HTML file for the frontend interface on the root path ('/').
    (This ensures the UI loads correctly).
    """
    try:
        # Assumes index.html is in the same directory as app.py
        return send_from_directory(app.root_path, 'index.html')
    except Exception as e:
        logging.error(f"Failed to serve index.html: {e}")
        return "ITSM RAG Service is running. Use the /query endpoint to submit questions. (Error serving index.html)", 500

@app.route('/query', methods=['POST'])
def rag_query():
    """Handles the RAG query API endpoint."""
    if not initialize_global_resources():
        return jsonify({"error": "Service initialization failed. Could not load required data."}), 500
        
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body."}), 400
        
        user_query = data['query']
        
        # Get the query embedding
        query_embedding_values = get_query_embedding(user_query)
        if query_embedding_values is None:
            return jsonify({"error": "Failed to generate query embedding."}), 500
            
        # Retrieve context using the new, detailed columns
        retrieved_incidents = retrieve_incidents_in_memory(query_embedding_values)
        rag_answer = generate_rag_answer(user_query, retrieved_incidents)
        
        # Check if the model's answer indicates a lack of information
        if "cannot answer" in rag_answer.lower() or "not sufficient" in rag_answer.lower():
            logging.info(f"RAG_NO_ANSWER_EVENT: Query='{user_query}'")
            
        return jsonify({"answer": rag_answer})
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during the RAG process: {e}")
        return jsonify({"error": f"An unexpected error occurred during the RAG process: {e}"}), 500

# --- Main Entry Point ---
if __name__ == '__main__':
    # Initialize data immediately when running locally
    initialize_global_resources()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
