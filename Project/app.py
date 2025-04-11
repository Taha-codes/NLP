import streamlit as st
import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import subprocess

# --- Utility Functions: Device, Index, and Embedding Functions ---

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_index(index_path="catalog_index.faiss"):
    return faiss.read_index(index_path)

def load_metadata(metadata_path="chunk_metadata.json"):
    with open(metadata_path, "r", encoding="utf-8") as infile:
        return json.load(infile)

def compute_query_embedding(query, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    device = get_device()
    model = SentenceTransformer(model_name, device=device)
    return model.encode([query], convert_to_numpy=True)

def retrieve_relevant_chunks(query, index, metadata, top_k=5):
    query_embedding = compute_query_embedding(query)
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = []
    for idx in indices[0]:
        if idx != -1 and idx < len(metadata):
            relevant_chunks.append(metadata[idx])
    return relevant_chunks

def construct_prompt(query, retrieved_chunks):
    prompt = f"User Question: {query}\n\n"
    prompt += ("Below are relevant excerpts from the AUI catalog. Please answer the question "
               "using only the context provided, and include citations based on the page numbers "
               "and section titles.\n")
    for chunk in retrieved_chunks:
        prompt += (
            f"\n[Catalog Page {chunk['page_number']} - {chunk['section_title']}]:\n"
            f"{chunk['chunk_text']}\n"
        )
    prompt += "\nAnswer the question based solely on the above context."
    return prompt

def generate_answer_with_ollama(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "deepseek-r1:8b"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        st.error("Error generating answer via Ollama:")
        st.error(e.stderr.decode("utf-8"))
        return None

# --- Streamlit Application ---

# Set page configuration
st.set_page_config(page_title="AUI Catalog Assistant", layout="wide")

st.title("AUI Catalog Assistant")
st.write("""
This tool uses a Retrieval-Augmented Generation (RAG) approach to answer your questions based on the AUI catalog. 
Type in your question below and receive an answer with citations!
""")

# Sidebar for settings (optional)
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of Context Chunks", min_value=3, max_value=10, value=5)
st.sidebar.write("Adjust the number of retrieved excerpts used as context.")

# Load FAISS index and metadata once at startup
@st.cache_resource(show_spinner="Loading index and metadata...")
def load_resources():
    index = load_index("catalog_index.faiss")
    metadata = load_metadata("chunk_metadata.json")
    return index, metadata

index, metadata = load_resources()

# User input for query
query = st.text_input("Enter your question regarding the AUI catalog:")

if st.button("Get Answer") and query:
    st.write("Processing your query...")
    retrieved_chunks = retrieve_relevant_chunks(query, index, metadata, top_k=top_k)
    
    if not retrieved_chunks:
        st.warning("No relevant catalog excerpts were found. Please rephrase your query.")
    else:
        # Display retrieved excerpts in an expander
        with st.expander("Show Retrieved Catalog Excerpts"):
            for chunk in retrieved_chunks:
                st.markdown(f"**Page {chunk['page_number']} - {chunk['section_title']}**")
                st.write(chunk['chunk_text'][:300] + "...")
        
        prompt = construct_prompt(query, retrieved_chunks)
        st.code(prompt, language="markdown")
        
        st.write("Generating answer using DeepSeek 8B...")
        answer = generate_answer_with_ollama(prompt)
        if answer:
            st.markdown("### Answer")
            st.write(answer)
        else:
            st.error("Failed to generate an answer.")