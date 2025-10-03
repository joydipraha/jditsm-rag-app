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
    """Downloads a blob from the bucket."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        logging.info(f"Downloading {source_blob_name} to {destination_file_name}...")
        blob.download_to_filename(destination_file_name)
        logging.info(f"Download complete: {destination_file_name}")
        return True
    except Exception as e:
        logging.error(f"Failed to download {source_blob_name}: {e}")
        return False

def load_data(csv_path, embeddings_path):
    """Loads CSV into DataFrame and JSON into NumPy array."""
    global INCIDENT_DATA
    global EMBEDDINGS_DATA
    
    try:
        # Load the CSV data
        INCIDENT_DATA = pd.read_csv(csv_path)
        logging.info(f"Loaded CSV data: {len(INCIDENT_DATA)} rows.")
        
        # Load the embeddings
        with open(embeddings_path, 'r') as f:
            embeddings_list = json.load(f)
            
        # Convert list of embeddings (list of lists) into a NumPy array
        EMBEDDINGS_DATA = np.array(embeddings_list)
        logging.info(f"Loaded embeddings data: {EMBEDDINGS_DATA.shape}")
        
    except Exception as e:
        logging.error(f"Failed to load local data files: {e}")
        INCIDENT_DATA = None
        EMBEDDINGS_DATA = None

# --- Initialization Function (Lazy Loading) ---
def initialize_global_resources():
    """
    Downloads and loads the required data files if they are not already loaded.
    This function uses a flag check for warm starts.
    """
    global INCIDENT_DATA
    global EMBEDDINGS_DATA
    
    # Check if data is already loaded (for warm start)
    if INCIDENT_DATA is not None and EMBEDDINGS_DATA is not None:
        logging.info("Global resources already loaded. Skipping initialization.")
        return True
    
    logging.info("Starting initial resource loading (cold start)...")

    # Use tempfile to ensure cleanup in /tmp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_local_path = os.path.join(temp_dir, CSV_FILE_NAME)
        embeddings_local_path = os.path.join(temp_dir, os.path.basename(EMBEDDINGS_FILE_NAME))

        # 1. Download CSV
        if not download_gcs_file(GCS_BUCKET_NAME, CSV_FILE_NAME, csv_local_path):
            logging.error("Initialization failed: Could not download CSV.")
            return False

        # 2. Download Embeddings
        if not download_gcs_file(GCS_BUCKET_NAME, EMBEDDINGS_FILE_NAME, embeddings_local_path):
            logging.error("Initialization failed: Could not download embeddings.")
            return False

        # 3. Load Data
        load_data(csv_local_path, embeddings_local_path)

    return INCIDENT_DATA is not None and EMBEDDINGS_DATA is not None

def get_query_embedding(query_text):
    """Generates the embedding for the user query."""
    try:
        response = embedding_model.embed_texts(
            texts=[query_text],
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768 # Match the model output dimension
        )
        return response[0].values
    except Exception as e:
        logging.error(f"Failed to generate query embedding: {e}")
        return None

def retrieve_incidents_in_memory(query_embedding_values, k=3):
    """
    Performs vector similarity search against the pre-loaded embeddings
    and constructs a detailed context string from all relevant columns,
    excluding the noisy 'DESCRIPTION'.
    """
    if EMBEDDINGS_DATA is None or INCIDENT_DATA is None:
        logging.error("Data not loaded for retrieval.")
        return []

    # Reshape the query embedding to be a 2D array for cosine_similarity
    query_vector = np.array(query_embedding_values).reshape(1, -1)
    
    # Calculate cosine similarity between the query vector and all incident vectors
    similarities = cosine_similarity(query_vector, EMBEDDINGS_DATA)[0]
    
    # Get the indices of the top K most similar incidents
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    # Retrieve the corresponding incident text/data
    retrieved_incidents = INCIDENT_DATA.iloc[top_k_indices]
    
    # CRITICAL FIX: Define only the valuable, structured columns for the LLM context.
    # The 'DESCRIPTION' field is explicitly excluded here.
    RAG_CONTEXT_COLUMNS = [
        'Incident ID', 
        'Reporter Name', 
        'Contact Type', 
        'Category', 
        'Item Affected (CI)', 
        'Short Description', 
        'Priority', 
        'Status', 
        'Assignment Group', 
        'SLA Breached', 
        'Root Cause', 
    ]

    # Create detailed context string for the LLM
    context = ""
    for index, row in retrieved_incidents.iterrows():
        # Start a structured block for each incident
        incident_context = f"--- INCIDENT RECORD START (Index: {index}) ---\n"
        
        for col in RAG_CONTEXT_COLUMNS:
            # Check if the column exists in the DataFrame (safe check)
            if col in row.index:
                # Format the column name and its value. Uppercasing and replacing spaces for LLM clarity.
                field_name = col.upper().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                # Ensure missing values are represented clearly, although this should be cleaned upstream.
                value = row[col] if pd.notna(row[col]) else "N/A" 
                incident_context += f"{field_name}: {value}\n"
            
        incident_context += f"--- INCIDENT RECORD END ---\n\n"
        context += incident_context
    
    return context

def generate_rag_answer(user_query, context):
    """
    Generates a final answer using the LLM based on the user query and retrieved context.
    """
    system_instruction = (
        "You are an ITSM Executive Assistant. Your task is to analyze the provided "
        "incident data (CONTEXT) which contains structured fields like Incident ID, Reporter Name, Category, "
        "Short Description, Priority, Status, and Root Cause. "
        "Answer the USER_QUERY concisely and professionally using ONLY the facts found in the CONTEXT. "
        "The original full 'DESCRIPTION' text was omitted due to containing demo data, so rely on the other structured fields."
        "If the CONTEXT does not contain sufficient information to answer the query, "
        "you MUST state that you 'cannot answer the question based on the provided incident data' and stop."
        "Do not invent information."
    )
    
    prompt = (
        f"USER_QUERY: {user_query}\n\n"
        f"CONTEXT (Incident Data):\n---\n{context}\n---"
    )
    
    try:
        response = llm.generate_content(
            contents=[prompt],
            config={"system_instruction": system_instruction}
        )
        return response.text
    except Exception as e:
        logging.error(f"LLM generation failed: {e}")
        return "Sorry, the AI model failed to generate an answer."

# --- Main Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/rag', methods=['POST'])
def rag_endpoint():
    """
    Handles the RAG search request, ensuring resources are loaded on first call.
    """
    # Ensure resources are loaded. This is called on every request, 
    # but the function itself handles the 'already loaded' case for warm starts.
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
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
