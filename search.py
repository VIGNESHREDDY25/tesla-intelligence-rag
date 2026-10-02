import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# Paths
INDEX_PATH = Path("data/faiss/tesla.index")
METADATA_PATH = Path("data/faiss/metadata.json")


# Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load FAISS index
index = faiss.read_index(str(INDEX_PATH))


# Load metadata
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


print("Search system ready!")


# Ask the user for a question
query = input("\nAsk a question about Tesla: ")


# Convert question into an embedding
query_embedding = model.encode([query])

# Convert to the format FAISS expects
query_embedding = query_embedding.astype("float32")

# Normalize the query vector
faiss.normalize_L2(query_embedding)


# Search for the 5 most similar chunks
k = 5

scores, indices = index.search(query_embedding, k)


# Display results
print("\n" + "=" * 80)
print("RETRIEVED CHUNKS")
print("=" * 80)


for rank, (score, index_id) in enumerate(
    zip(scores[0], indices[0]), start=1
):

    result = metadata[index_id]

    print(f"\n--- Result {rank} ---")
    print(f"Similarity score: {score:.4f}")
    print(f"Year: {result['year']}")
    print(f"Page: {result['page']}")
    print(f"Source: {result['source']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print("\nText:")
    print(result["text"])


print("\n" + "=" * 80)