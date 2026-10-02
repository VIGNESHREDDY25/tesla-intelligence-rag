import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


# Paths
INPUT_DIR = Path("data/chunks_v3")
OUTPUT_DIR = Path("data/embeddings")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")


# Process each Tesla year
for input_file in sorted(INPUT_DIR.glob("*_chunks.json")):

    print(f"\nProcessing: {input_file.name}")

    # Read chunks
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Found {len(chunks)} chunks")

    # Extract text
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    # Combine embeddings with metadata
    output_data = []

    for chunk, embedding in zip(chunks, embeddings):

        output_data.append({
            "chunk_id": chunk["chunk_id"],
            "year": chunk["year"],
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "embedding": embedding.tolist()
        })

    # Output filename
    output_file = OUTPUT_DIR / input_file.name.replace(
        "_chunks.json",
        "_embeddings.json"
    )

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f)

    print(f"Saved: {output_file}")
    print(f"Embeddings created: {len(output_data)}")


print("\nAll embeddings created successfully!")