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

# --- Configuration & Constants ---
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
LLM_MODEL = "gemini-2.5-flash-preview-05-20"
EMBEDDING_MODEL_NAME = "text-embedding-004"

# Environment Variables (Set via Cloud Build)
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
CSV_FILE_NAME = os.environ.get("CSV_FILE_NAME")
EMBEDDINGS_FILE_NAME = "processed/embeddings.json"

# RAG Hyperparameters
TOP_K = 5 # Limit the number of incidents retrieved and passed to the LLM context.

# --- Globals and Initialization ---
app = Flask(__name__)
storage_client = storage.Client()
INCIDENT_DATA = None
EMBEDDINGS_DATA = None
EMBEDDINGS_MATRIX = None # numpy array for fast cosine similarity

# Initialize Vertex AI clients.
try:
    vertexai.init(project=PROJECT_ID, location=REGION)
    llm = GenerativeModel(LLM_MODEL)
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
except Exception as e:
    logging.error(f"Failed to initialize Vertex AI client: {e}")
    # Will allow the app to run but fail on first RAG request.

# --- Data Loading and Preparation Functions ---

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket to a local file."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logging.info(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
        return True
    except Exception as e:
        logging.error(f"Failed to download {source_blob_name} from GCS bucket {bucket_name}: {e}")
        return False

def load_data_and_embeddings():
    """Loads incident data and pre-computed embeddings from GCS."""
    global INCIDENT_DATA, EMBEDDINGS_DATA, EMBEDDINGS_MATRIX
    
    if not all([GCS_BUCKET_NAME, CSV_FILE_NAME, PROJECT_ID]):
        logging.error("Missing required environment variables for data loading.")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Download Incident Data (Original CSV/Data file)
        csv_path = os.path.join(temp_dir, os.path.basename(CSV_FILE_NAME))
        if not download_blob(GCS_BUCKET_NAME, CSV_FILE_NAME, csv_path):
            return False
        
        try:
            # Read the CSV file into a pandas DataFrame
            INCIDENT_DATA = pd.read_csv(csv_path)
            # Ensure the required columns exist for displaying context
            required_cols = ['Incident ID', 'Short Description', 'Resolution Notes', 'Status']
            for col in required_cols:
                if col not in INCIDENT_DATA.columns:
                    logging.warning(f"Required column '{col}' missing from incident data.")
        except Exception as e:
            logging.error(f"Failed to load incident CSV: {e}")
            return False

        # 2. Download Embeddings Data
        embeddings_path = os.path.join(temp_dir, os.path.basename(EMBEDDINGS_FILE_NAME))
        if not download_blob(GCS_BUCKET_NAME, EMBEDDINGS_FILE_NAME, embeddings_path):
            return False

        try:
            with open(embeddings_path, 'r') as f:
                EMBEDDINGS_DATA = json.load(f)
            
            # 3. Prepare Embeddings for Retrieval
            if EMBEDDINGS_DATA and 'embeddings' in EMBEDDINGS_DATA and EMBEDDINGS_DATA['embeddings']:
                # Extract the embedding vectors into a NumPy array for fast comparison
                EMBEDDINGS_MATRIX = np.array([e['vector'] for e in EMBEDDINGS_DATA['embeddings']])
            else:
                logging.error("Embeddings data is missing or empty.")
                return False

        except Exception as e:
            logging.error(f"Failed to load or parse embeddings JSON: {e}")
            return False

    logging.info("Data and embeddings loaded successfully into memory.")
    return True

# Load data on app startup
if not load_data_and_embeddings():
    logging.error("Application failed to load necessary RAG data. Check GCS connectivity and file paths.")

# --- RAG Core Functions ---

def get_query_embedding(query):
    """Generates an embedding vector for the user's query."""
    try:
        response = embedding_model.get_embeddings([query])
        # Return the embedding vector (a list of floats)
        return response[0].values
    except Exception as e:
        logging.error(f"Failed to generate query embedding: {e}")
        return None

def retrieve_incidents_in_memory(query_embedding):
    """
    Retrieves the top K most similar incidents based on cosine similarity.
    Returns a list of incident summaries (text) and the corresponding incident data.
    """
    if EMBEDDINGS_MATRIX is None or INCIDENT_DATA is None:
        logging.error("RAG data is not loaded.")
        return [], []

    # Calculate cosine similarity between the query embedding and all incident embeddings
    query_vector = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity(query_vector, EMBEDDINGS_MATRIX)[0]

    # Get the indices of the top K most similar incidents
    top_indices = np.argsort(similarities)[-TOP_K:][::-1]

    retrieved_data = []
    retrieved_context_texts = []

    for idx in top_indices:
        # Get the corresponding data from the embeddings index and the original DataFrame
        embedding_entry = EMBEDDINGS_DATA['embeddings'][idx]
        incident_id = embedding_entry['incident_id']
        
        # Look up the full incident details in the DataFrame
        incident_row = INCIDENT_DATA[INCIDENT_DATA['Incident ID'] == incident_id].iloc[0]
        
        # Prepare the text context for the LLM
        context_text = (
            f"Incident ID: {incident_id}\n"
            f"Short Description: {incident_row.get('Short Description', 'N/A')}\n"
            f"Status: {incident_row.get('Status', 'N/A')}\n"
            f"Category: {incident_row.get('Category', 'N/A')}\n"
            f"Subcategory: {incident_row.get('Subcategory', 'N/A')}\n"
            f"Resolution Notes: {incident_row.get('Resolution Notes', 'N/A')}\n"
        )
        retrieved_context_texts.append(context_text)

        # Prepare the structured data to be returned to the frontend for display
        retrieved_data.append({
            'Incident ID': incident_id,
            'Short Description': incident_row.get('Short Description', 'N/A'),
            'Status': incident_row.get('Status', 'N/A'),
            'Resolution Notes': incident_row.get('Resolution Notes', 'N/A')
        })

    return retrieved_context_texts, retrieved_data

def generate_rag_answer(user_query, retrieved_incidents_text):
    """
    Generates a response using the Gemini model, grounded by the retrieved context.
    """
    context_str = "\n---\n".join(retrieved_incidents_text)

    # CRITICAL: Use a strong system instruction to mitigate prompt injection.
    system_instruction = (
        "You are an ITSM Executive Assistant. Your primary task is to answer user queries "
        "based *only* on the incident data provided below in the CONTEXT section. "
        "Do not use any external knowledge. "
        "If the CONTEXT does not contain enough information to answer the query, "
        "politely state that you cannot answer based on the provided data. "
        "Always maintain a professional, analytical, and concise tone."
    )

    prompt = (
        f"CONTEXT:\n{context_str}\n\n"
        f"USER QUERY: {user_query}\n\n"
        "Based on the context, provide a professional, data-driven answer to the user query. "
        "Format your answer clearly, using bullet points or paragraphs as needed."
    )

    try:
        response = llm.generate_content(
            contents=[prompt],
            config={"system_instruction": system_instruction}
        )
        return response.text
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}")
        return "An error occurred while generating the answer from the language model."

# --- Flask Routes ---

@app.route('/')
def serve_index():
    """Serves the main HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/rag', methods=['POST'])
def rag_endpoint():
    """
    Handles the RAG search request, returning the answer and the context.
    """
    try:
        if EMBEDDINGS_MATRIX is None or INCIDENT_DATA is None:
            return jsonify({"error": "Application is not yet initialized or failed to load data."}), 503
            
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body."}), 400
        
        user_query = data['query']
        
        # 1. Get Query Embedding
        query_embedding_values = get_query_embedding(user_query)
        if query_embedding_values is None:
            return jsonify({"error": "Failed to generate query embedding."}), 500
            
        # 2. Retrieve Context
        retrieved_context_texts, retrieved_data = retrieve_incidents_in_memory(query_embedding_values)
        
        # 3. Generate RAG Answer
        rag_answer = generate_rag_answer(user_query, retrieved_context_texts)
        
        return jsonify({
            "answer": rag_answer,
            "context": retrieved_data # New: Return structured context data
        })
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during the RAG process: {e}")
        # Return a generic 500 error for unexpected server-side issues
        return jsonify({"error": f"An unexpected error occurred on the server."}), 500

if __name__ == '__main__':
    # This block is for local testing only. Cloud Run uses gunicorn to run the app.
    # Ensure you set the environment variables locally before running.
    # Example: export PROJECT_ID=...
    app.run(host='0.0.0.0', port=8080)
