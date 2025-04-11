import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def load_catalog_chunks(json_path="catalog_chunks.json"):
    with open(json_path, "r", encoding="utf-8") as infile:
        catalog_chunks = json.load(infile)
    return catalog_chunks

def get_device():
    """
    Check for the availability of the MPS backend on macOS.
    If available, use it; otherwise, default to CPU.
    """
    if torch.backends.mps.is_available():
        print("Using MPS device for embedding computations.")
        return "mps"
    else:
        print("MPS not available. Using CPU.")
        return "cpu"

def compute_embeddings(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Compute embeddings for each text chunk using a pre-trained MiniLM model.
    
    Args:
        chunks (list): List of dictionaries, each containing a "chunk_text" field.
        model_name (str): Hugging Face model ID to use for embeddings.
        
    Returns:
        np.array: Numpy array of embeddings.
    """
    device = get_device()
    # Load the model on the specified device
    model = SentenceTransformer(model_name, device=device)
    
    documents = [chunk['chunk_text'] for chunk in chunks]
    embeddings = model.encode(documents, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings):
    """
    Build a FAISS index from the embeddings.
    
    Args:
        embeddings (np.array): Numpy array of shape (num_chunks, embedding_dim).
        
    Returns:
        faiss.Index: A FAISS index containing the embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_index(index, index_path="catalog_index.faiss"):
    faiss.write_index(index, index_path)
    
def main():
    # Step 1: Load the segmented chunks from JSON.
    catalog_chunks = load_catalog_chunks("catalog_chunks.json")
    print(f"Loaded {len(catalog_chunks)} chunks.")

    # Step 2: Compute embeddings for each chunk using the MiniLM model, on GPU if available.
    print("Computing embeddings...")
    embeddings = compute_embeddings(catalog_chunks)
    print(f"Computed embeddings for {embeddings.shape[0]} chunks with dimension {embeddings.shape[1]}.")

    # Step 3: Build a FAISS index.
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # Step 4: Save the FAISS index and chunk metadata mapping.
    save_index(index, "catalog_index.faiss")
    with open("chunk_metadata.json", "w", encoding="utf-8") as outfile:
        json.dump(catalog_chunks, outfile, indent=4, ensure_ascii=False)
    
    print("Index and metadata have been saved successfully.")

if __name__ == "__main__":
    main()
    