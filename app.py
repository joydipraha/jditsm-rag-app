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
from google.cloud import storage
import threading 

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
# Read values from environment variables.
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "text-embedding-004"

# These must be set as environment variables during deployment.
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
CSV_FILE_NAME = os.environ.get("CSV_FILE_NAME")
# This file path is hardcoded but relies on the GCS_BUCKET_NAME being correct
EMBEDDINGS_FILE_NAME = "processed/embeddings.json" 

# --- Globals and Initialization ---\
app = Flask(__name__)
# The storage_client is initialized globally, which is fine
storage_client = storage.Client() 
INCIDENT_DATA = None
EMBEDDINGS_DATA = None
llm = None
embedding_model = None
# Lock for thread-safe initialization, essential for Cloud Run
initialization_lock = threading.Lock() 

# --- Utility Functions ---

def load_data_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket and saves it locally."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        # CRUCIAL CHECK: Log if the file is missing in GCS
        if not blob.exists():
            logging.error(f"GCS Error: Blob '{source_blob_name}' does not exist. Check file path or bucket permissions.")
            return False

        blob.download_to_filename(destination_file_name)
        logging.info(f"Successfully downloaded {source_blob_name} to {destination_file_name}.")
        return True
    except Exception as e:
        # Log specific GCS errors (e.g., permission denied)
        logging.error(f"Error loading data from GCS for '{source_blob_name}': {type(e).__name__} - {e}")
        return False

def get_query_embedding(text: str) -> np.ndarray:
    """Generates an embedding for the given text."""
    global embedding_model
    try:
        if embedding_model is None:
            logging.error("Embedding model is not initialized.")
            return None
            
        embedding_object = embedding_model.get_embeddings([text])[0]
        return np.array(embedding_object.values, dtype=np.float32)
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

def retrieve_incidents_in_memory(query_embedding: np.ndarray, top_k: int = 5):
    """
    Performs a vector similarity search and retrieves all 11 required fields for RAG context.
    
    The default top_k is 5.
    """
    global EMBEDDINGS_DATA, INCIDENT_DATA

    if EMBEDDINGS_DATA is None or INCIDENT_DATA is None:
        logging.error("RAG Data (INCIDENT_DATA or EMBEDDINGS_DATA) is not loaded.")
        return []

    # 1. Prepare data for cosine similarity calculation
    incident_embeddings = np.array([item['embedding'] for item in EMBEDDINGS_DATA], dtype=np.float32)
    query_embedding_reshaped = query_embedding.reshape(1, -1)

    # 2. Calculate Cosine Similarity
    similarity_scores = cosine_similarity(query_embedding_reshaped, incident_embeddings)[0]

    # 3. Get top K indices
    top_k_indices = np.argsort(similarity_scores)[-top_k:][::-1]

    # 4. Retrieve the actual incident data and format
    retrieved_incidents = []
    
    # Map the desired descriptive field name to the column name in the DataFrame
    FIELD_MAPPING = {
        "Incident ID": "number",
        "Reporter Name": "caller_id",
        "Contact Type": "contact_type",
        "Category": "category",
        "Item Affected (CI)": "cmdb_ci",
        "Short Description": "short_description",
        "Priority": "priority",
        "Status": "incident_state", 
        "Assignment Group": "assignment_group",
        "SLA Breached": "sla_due", 
        "Root Cause": "root_cause",
    }
    
    for idx in top_k_indices:
        incident_row = INCIDENT_DATA.iloc[idx].to_dict()
        
        incident_context = {}
        for descriptive_name, column_name in FIELD_MAPPING.items():
            # Get data, falling back to "N/A" if column or value is missing
            value = incident_row.get(column_name)
            incident_context[descriptive_name] = str(value) if pd.notna(value) else "N/A"
        
        retrieved_incidents.append(incident_context)

    return retrieved_incidents

def generate_rag_answer(user_query, retrieved_incidents):
    """Generates an answer using Gemini, grounded on the retrieved context."""
    global llm

    if llm is None:
        logging.error("LLM model is not initialized.")
        return "Service initialization error: LLM is not ready."

    # Format the detailed context for the LLM, using the descriptive names
    context_text = []
    for inc in retrieved_incidents:
        context_block = "--- Incident Start ---\n"
        for key, value in inc.items():
            context_block += f"{key}: {value}\n"
        context_block += "--- Incident End ---\n"
        context_text.append(context_block)
        
    formatted_context = "\n".join(context_text)

    prompt = f"""
    You are an expert ITSM (IT Service Management) Executive Assistant. Your goal is to answer a user's question concisely based ONLY on the provided relevant incident context. You can synthesize information across multiple incidents if necessary.

    The context contains the following key fields for each incident: Incident ID, Reporter Name, Contact Type, Category, Item Affected (CI), Item Affected (CI), Short Description, Priority, Status, Assignment Group, SLA Breached, and Root Cause.

    If the context does not contain sufficient information to answer the question, you must clearly state that you "cannot answer based on the provided incidents."

    **Incident Context (Use these fields for grounding):**
    {formatted_context}

    **User Question:**
    {user_query}

    **Your Answer:**
    """

    try:
        # Fix: The keyword 'config' was replaced with 'generation_config'
        response = llm.generate_content(
            prompt,
            generation_config={"temperature": 0.2}
        )
        return response.text
    except Exception as e:
        # Log the specific error if possible, but return a generic message to the frontend
        logging.error(f"Error calling LLM for RAG answer: {e}")
        return "Failed to generate answer due to an LLM service error."


# --- Global Resource Initialization ---

def initialize_global_resources():
    """Initializes global resources (models and data) in a thread-safe manner."""
    global INCIDENT_DATA, EMBEDDINGS_DATA, llm, embedding_model, initialization_lock
    
    # Check if already initialized (fast path)
    if INCIDENT_DATA is not None and EMBEDDINGS_DATA is not None:
        return True

    with initialization_lock:
        # Double-check inside the lock
        if INCIDENT_DATA is not None and EMBEDDINGS_DATA is not None:
            return True

        logging.info("Starting global resource initialization...")

        # --- 0. Check Environment Variables ---
        required_env_vars = ["PROJECT_ID", "REGION", "GCS_BUCKET_NAME", "CSV_FILE_NAME"]
        missing_vars = [v for v in required_env_vars if not os.environ.get(v)]
        
        if missing_vars:
            logging.error(f"FATAL: Missing required environment variables: {', '.join(missing_vars)}")
            return False

        # --- 1. Initialize Vertex AI Clients (LLM and Embedding) ---
        try:
            vertexai.init(project=PROJECT_ID, location=REGION)
            llm = GenerativeModel(LLM_MODEL)
            embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
            logging.info("Vertex AI models initialized successfully.")
        except Exception as e:
            logging.error(f"FATAL: Error initializing Vertex AI models. This may indicate permission issues or incorrect region/project ID: {e}")
            return False

        # --- 2. Load Incident Data (CSV) ---
        try:
            logging.info(f"Attempting to download incident data from GCS bucket: '{GCS_BUCKET_NAME}', file: '{CSV_FILE_NAME}'")
            if not load_data_from_gcs(GCS_BUCKET_NAME, CSV_FILE_NAME, 'incident_data.csv'):
                logging.error("Failed to load incident CSV data from GCS.")
                return False
            
            # Read CSV and ensure we handle potential missing columns gracefully if data structure changes
            INCIDENT_DATA = pd.read_csv('incident_data.csv')
            logging.info(f"Loaded {len(INCIDENT_DATA)} incident records.")
        except Exception as e:
            logging.error(f"FATAL: Error processing incident CSV data (read_csv failure): {e}")
            return False

        # --- 3. Load Embeddings Data (JSON) ---
        try:
            logging.info(f"Attempting to download embeddings data from GCS bucket: '{GCS_BUCKET_NAME}', file: '{EMBEDDINGS_FILE_NAME}'")
            if not load_data_from_gcs(GCS_BUCKET_NAME, EMBEDDINGS_FILE_NAME, 'embeddings.json'):
                logging.error("Failed to load embeddings data from GCS.")
                return False
            
            with open('embeddings.json', 'r') as f:
                EMBEDDINGS_DATA = json.load(f)
            logging.info(f"Loaded {len(EMBEDDINGS_DATA)} embedding vectors.")
        except Exception as e:
            logging.error(f"FATAL: Error processing embeddings JSON data (json.load failure): {e}")
            return False
        
        logging.info("Global resource initialization COMPLETE.")
        return True

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Serves the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/rag', methods=['POST'])
def rag():
    """Handles the RAG query."""
    # Ensure all resources are loaded before handling a request
    if not initialize_global_resources():
        # This will return the error to the frontend if initialization failed
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
            
        # Retrieve context
        # top_k remains 5 as per user request
        retrieved_incidents = retrieve_incidents_in_memory(query_embedding_values, top_k=5)
        
        # Generate RAG answer
        rag_answer = generate_rag_answer(user_query, retrieved_incidents)
        
        # We only send back the ID and Short Description for the UI to display as context links,
        # but the full context was used in the prompt for the answer.
        display_context = [
            {"incident_id": item['Incident ID'], "short_description": item['Short Description']} 
            for item in retrieved_incidents
        ]
        
        return jsonify({
            "answer": rag_answer, 
            "context": display_context
        })
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during the RAG process: {e}")
        return jsonify({"error": f"An unexpected error occurred during the RAG process: {e}"}), 500

# --- Main Entry Point ---

if __name__ == '__main__':
    if initialize_global_resources():
        # Using 0.0.0.0 and PORT from environment is standard for Cloud Run
        app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
    else:
        logging.error("Application failed to start due to resource initialization error.")
