import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def load_index(index_path="catalog_index.faiss"):
    index = faiss.read_index(index_path)
    return index

def load_metadata(metadata_path="chunk_metadata.json"):
    with open(metadata_path, "r", encoding="utf-8") as infile:
        metadata = json.load(infile)
    return metadata

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def compute_query_embedding(query, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    device = get_device()
    model = SentenceTransformer(model_name, device=device)
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding

def retrieve_relevant_chunks(query, index, metadata, top_k=5):
    # Compute embedding for the query
    query_embedding = compute_query_embedding(query)
    
    # Perform search in the FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    # Get the top relevant chunks using the indices
    relevant_chunks = []
    for idx in indices[0]:
        if idx != -1 and idx < len(metadata):
            relevant_chunks.append(metadata[idx])
    return relevant_chunks

def construct_prompt(query, retrieved_chunks):
    """
    Build a prompt for the LLM by including the user query and the retrieved catalog chunks.
    Each chunk includes its section title for citation reference.
    """
    prompt = f"User Question: {query}\n\n"
    prompt += "Relevant Catalog Excerpts (please include citations in your answer):\n"
    for chunk in retrieved_chunks:
        prompt += f"\n[Catalog Page {chunk['page_number']} - {chunk['section_title']}]:\n{chunk['chunk_text']}\n"
    prompt += "\nAnswer the question based solely on the above context."
    return prompt

def main():
    # Load FAISS index and metadata
    index = load_index("catalog_index.faiss")
    metadata = load_metadata("chunk_metadata.json")
    
    # Example query from a user
    query = "What are the prerequisites for the Software Engineering program?"
    print("User Query:", query)
    
    # Retrieve top K relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(query, index, metadata, top_k=5)
    print("Retrieved Chunks:")
    for chunk in retrieved_chunks:
        print(f"Page {chunk['page_number']} - {chunk['section_title']}: {chunk['chunk_text'][:100]}...")
    
    # Construct prompt for LLM (DeepSeek 8B)
    prompt = construct_prompt(query, retrieved_chunks)
    print("\nConstructed Prompt for LLM:\n", prompt)
    
    # Next: Use this prompt as input to your DeepSeek 8B model to generate an answer
    # For example:
    # answer = deepseek_generate(prompt)
    # print("\nAnswer:", answer)

if __name__ == "__main__":
    main()