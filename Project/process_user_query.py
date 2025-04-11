import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import subprocess

#########################
# Device Setup
#########################
def get_device():
    """
    Check for the availability of the MPS backend on macOS.
    Returns 'mps' if available; otherwise 'cpu'.
    """
    if torch.backends.mps.is_available():
        print("Using MPS device for computations.")
        return "mps"
    else:
        print("MPS not available. Using CPU.")
        return "cpu"

#########################
# FAISS Index and Metadata Loading
#########################
def load_index(index_path="catalog_index.faiss"):
    """Load the FAISS index from disk."""
    index = faiss.read_index(index_path)
    return index

def load_metadata(metadata_path="chunk_metadata.json"):
    """Load chunk metadata (text chunks with page numbers and section titles) from a JSON file."""
    with open(metadata_path, "r", encoding="utf-8") as infile:
        metadata = json.load(infile)
    return metadata

#########################
# Query Embedding and Retrieval
#########################
def compute_query_embedding(query, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Compute an embedding for the given query using MiniLM.
    """
    device = get_device()
    model = SentenceTransformer(model_name, device=device)
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding

def retrieve_relevant_chunks(query, index, metadata, top_k=5):
    """
    Retrieve the top k text chunks (with metadata) relevant to the user query.
    """
    query_embedding = compute_query_embedding(query)
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = []
    for idx in indices[0]:
        if idx != -1 and idx < len(metadata):
            relevant_chunks.append(metadata[idx])
    return relevant_chunks

def construct_prompt(query, retrieved_chunks):
    """
    Build a prompt for the language model using the query and the retrieved catalog excerpts.
    """
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

#########################
# Answer Generation via Ollama
#########################
def generate_answer_with_ollama(prompt):
    """
    Generate an answer by passing the prompt to DeepSeek 8B via Ollama.
    It calls: `ollama run deepseek-r1:8b` and passes the prompt via standard input.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "deepseek-r1:8b"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        output_text = result.stdout.decode("utf-8")
        return output_text
    except subprocess.CalledProcessError as e:
        print("Error generating answer via ollama:")
        print(e.stderr.decode("utf-8"))
        return None

#########################
# Main Pipeline Integration
#########################
def main():
    # Step 1: Load FAISS index and metadata from disk.
    index = load_index("catalog_index.faiss")
    metadata = load_metadata("chunk_metadata.json")
    print(f"Loaded FAISS index with {index.ntotal} vectors and metadata for {len(metadata)} chunks.\n")
    
    # Step 2: Get the user query.
    query = input("Enter your query: ")
    print(f"\nUser Query: {query}\n")
    
    # Step 3: Retrieve the most relevant catalog chunks.
    retrieved_chunks = retrieve_relevant_chunks(query, index, metadata, top_k=5)
    if not retrieved_chunks:
        print("No relevant catalog excerpts found.")
        return
    print("Retrieved Catalog Excerpts:")
    for chunk in retrieved_chunks:
        preview = chunk['chunk_text'][:100].replace('\n', ' ')  # first 100 characters as preview
        print(f"Page {chunk['page_number']} - {chunk['section_title']}: {preview}...")
    
    # Step 4: Construct the prompt for DeepSeek.
    prompt = construct_prompt(query, retrieved_chunks)
    print("\nConstructed Prompt:\n", prompt)
    
    # Step 5: Generate the answer using your pre-downloaded DeepSeek 8B model via Ollama.
    print("\nGenerating answer using ollama run deepseek-r1:8b...")
    answer = generate_answer_with_ollama(prompt)
    if answer:
        print("\nGenerated Answer:\n", answer)
    else:
        print("Failed to generate answer with DeepSeek 8B via Ollama.")

if __name__ == "__main__":
    main()