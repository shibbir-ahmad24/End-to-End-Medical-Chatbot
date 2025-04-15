from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import numpy as np
import faiss
import pickle

# Load and process data
print("[INFO] Loading PDF documents and splitting into chunks...")
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Convert text_chunks to a list of strings
text_strings = [str(chunk) for chunk in text_chunks]

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Get embedding dimension (same)
embedding_example = embeddings.embed_query("This is a test query.")
embedding_dimension = len(embedding_example) 

# Create a Faiss index
print("[INFO] Creating a Faiss index...")
index = faiss.IndexFlatL2(embedding_dimension)

# Store embeddings into Faiss index
print(f"[INFO] Storing embeddings into Faiss index...")
vectors = np.array(embeddings.embed_documents(text_strings), dtype=np.float32)
index.add(vectors)

print("[SUCCESS] Embeddings successfully stored in Faiss.")

# Save both Faiss index and text_strings
with open('faiss_index.pkl', 'wb') as f:
    pickle.dump((index, text_strings), f)

print("[SUCCESS] Faiss index and text strings saved to faiss_index.pkl")

# To search for similar vectors
def search(query_vector, k=5):
 D, I = index.search(np.array([query_vector], dtype=np.float32), k=k)
 return I[0], D[0]

query_vector = embeddings.embed_query("This is a test query.")
similar_indices, distances = search(query_vector)

print("Indices of similar vectors:", similar_indices)
print("Distances to similar vectors:", distances)

# To get the text chunks corresponding to the similar indices
similar_text_chunks = [text_strings[i] for i in similar_indices]
print("Similar text chunks:", similar_text_chunks)