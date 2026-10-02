import json
from pathlib import Path

import faiss
import numpy as np


# Paths
INPUT_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("data/faiss")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Store all embeddings and metadata
all_embeddings = []
all_metadata = []


# Load embedding files
for input_file in sorted(INPUT_DIR.glob("*_embeddings.json")):

    print(f"Loading: {input_file.name}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        all_embeddings.append(item["embedding"])

        all_metadata.append({
            "chunk_id": item["chunk_id"],
            "year": item["year"],
            "source": item["source"],
            "page": item["page"],
            "text": item["text"]
        })


print(f"\nTotal embeddings: {len(all_embeddings)}")


# Convert to NumPy array
embeddings = np.array(all_embeddings, dtype="float32")

print(f"Embedding shape: {embeddings.shape}")


# Normalize vectors
faiss.normalize_L2(embeddings)


# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


print(f"FAISS index contains: {index.ntotal} vectors")


# Save FAISS index
index_path = OUTPUT_DIR / "tesla.index"

faiss.write_index(index, str(index_path))


# Save metadata separately
metadata_path = OUTPUT_DIR / "metadata.json"

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False)


print(f"\nFAISS index saved to: {index_path}")
print(f"Metadata saved to: {metadata_path}")
print("\nFAISS index built successfully!")