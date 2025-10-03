import pandas as pd
import os
import json
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, Content
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
    # FIX: Initialize GenerativeModel directly.
    llm = GenerativeModel(LLM_MODEL)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
except Exception as e:
    logging.error(f"Failed to initialize Vertex AI services: {e}")
    # This exception handling is important for Cloud Run startup failures.

def load_data_from_gcs():
    """Downloads incident data (CSV) and embeddings (JSON) from GCS."""
    global INCIDENT_DATA, EMBEDDINGS_DATA
    
    # 1. Download CSV data
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(CSV_FILE_NAME)
        
        # Use a temporary file to download the CSV
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_csv:
            blob.download_to_filename(temp_csv.name)
            temp_csv.seek(0)
            INCIDENT_DATA = pd.read_csv(temp_csv.name)
        logging.info(f"Successfully loaded {len(INCIDENT_DATA)} incident records.")
    except Exception as e:
        logging.error(f"Error loading CSV data from GCS: {e}")
        return False

    # 2. Download embeddings data
    try:
        blob = bucket.blob(EMBEDDINGS_FILE_NAME)
        embeddings_json = blob.download_as_text()
        EMBEDDINGS_DATA = json.loads(embeddings_json)
        logging.info(f"Successfully loaded {len(EMBEDDINGS_DATA)} embeddings.")
    except Exception as e:
        logging.error(f"Error loading embeddings JSON from GCS: {e}")
        return False
        
    return True

def initialize_global_resources():
    """Initializes global data structures if they haven't been loaded."""
    global INCIDENT_DATA, EMBEDDINGS_DATA
    
    if INCIDENT_DATA is None or EMBEDDINGS_DATA is None:
        return load_data_from_gcs()
    return True

# --- Embedding and Retrieval Logic ---

def get_query_embedding(query: str):
    """Generates the embedding for the user's query."""
    try:
        embedding = embedding_model.get_embeddings([query])
        # Return only the vector list
        return embedding[0].values
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        return None

def retrieve_incidents_in_memory(query_embedding: list, top_k: int = 5):
    """Performs cosine similarity search against all stored embeddings."""
    if EMBEDDINGS_DATA is None:
        logging.error("Embeddings data is not loaded.")
        return []

    # Convert embeddings list of lists to a numpy array for efficient computation
    document_embeddings = np.array([item['embedding'] for item in EMBEDDINGS_DATA])
    
    # Compute similarity between the query and all document embeddings
    # Reshape query_embedding to be a 2D array (1, n_features) for similarity calculation
    query_vector = np.array(query_embedding).reshape(1, -1)
    
    # Cosine similarity returns a 1D array of scores
    similarity_scores = cosine_similarity(query_vector, document_embeddings)[0]
    
    # Get the indices of the top_k scores (most relevant)
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    
    retrieved_incidents = []
    
    # Define the required columns for the RAG context
    REQUIRED_COLUMNS = [
        'Incident ID', 'Reporter Name', 'Contact Type', 'Category', 
        'Item Affected (CI)', 'Short Description', 'Priority', 'Status', 
        'Assignment Group', 'SLA Breached', 'Root Cause'
    ]
    
    for idx in top_indices:
        # Get the row from the incident data
        incident_row = INCIDENT_DATA.iloc[idx]
        
        # Construct a readable context string using the required columns
        context_parts = []
        for col in REQUIRED_COLUMNS:
            # Check if the column exists to prevent key errors if data schema changed
            if col in incident_row:
                context_parts.append(f"{col}: {incident_row[col]}")
            else:
                logging.warning(f"Column '{col}' not found in incident data.")

        context = ", ".join(context_parts)
        retrieved_incidents.append(context)
        
    return retrieved_incidents

# --- LLM Generation ---

def generate_rag_answer(query: str, context: list) -> str:
    """Generates a final answer using the LLM based on retrieved context."""
    if not context:
        return f"Based on the available incident data, I cannot answer the query: '{query}'."

    context_str = "\n---\n".join(context)
    
    prompt = f"""
    You are an expert IT Service Management (ITSM) analyst. Your goal is to answer a user's question based ONLY on the provided relevant incident records below.

    Context/Incident Records:
    ---
    {context_str}
    ---

    User Query: "{query}"

    Instructions:
    1. Base your answer strictly on the provided Context/Incident Records.
    2. If the context does not contain sufficient information to answer the query, state clearly: "Based on the available incident data, I cannot answer this query."
    3. Be concise and focus on combining the information from the incidents to give a useful summary.
    """
    
    try:
        # Use the globally defined llm instance
        response = llm.generate_content(
            contents=prompt
        )
        return response.text
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}")
        return f"An error occurred while generating the answer: {e}"

# --- Flask Routes ---

@app.route('/')
def home():
    """A simple homepage for health checks and status."""
    return "ITSM RAG Service is running. Use the /query endpoint to submit questions."

@app.route('/query', methods=['POST'])
def query():
    """The main RAG endpoint."""
    
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
    # Initialize data immediately for local testing, otherwise Cloud Run handles it on first request
    initialize_global_resources()
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
