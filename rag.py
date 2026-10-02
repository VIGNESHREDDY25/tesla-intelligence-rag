import json
import os

import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# 1. Load FAISS index
# -----------------------------

index = faiss.read_index("data/faiss/tesla.index")

with open("data/faiss/metadata.json", "r") as f:
    metadata = json.load(f)


# -----------------------------
# 2. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 3. Connect to Groq
# -----------------------------

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# 4. Get user's question
# -----------------------------

query = input("\nAsk a question about Tesla: ")


# -----------------------------
# 5. Convert question to embedding
# -----------------------------

query_embedding = model.encode([query])
query_embedding = query_embedding.astype("float32")

faiss.normalize_L2(query_embedding)


# -----------------------------
# 6. Retrieve relevant chunks
# -----------------------------

k = 5

scores, indices = index.search(query_embedding, k)


# -----------------------------
# 7. Build context
# -----------------------------

context = ""

for rank, idx in enumerate(indices[0], start=1):

    item = metadata[idx]

    context += f"""
Source {rank}:
Year: {item['year']}
Page: {item['page']}
Chunk ID: {item['chunk_id']}

Text:
{item['text']}

-------------------------
"""


# -----------------------------
# 8. Create RAG prompt
# -----------------------------

prompt = f"""
You are a Tesla company research assistant.

Answer the user's question using ONLY the information provided
in the context below.

If the context does not contain enough information to answer
the question, say that the available Tesla documents do not
contain enough information.

Do not make up facts.

After your answer, provide a "Sources" section.

For each source you actually used, include:
- Tesla document year
- Page number
- Chunk ID

Use this format:

Answer:
<your answer>

Sources:
- Tesla <year> 10-K — Page <page> — Chunk <chunk_id>

User question:
{query}

Context:
{context}
"""


# -----------------------------
# 9. Send prompt to Groq
# -----------------------------

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
)


# -----------------------------
# 10. Get final answer
# -----------------------------

answer = response.choices[0].message.content


# -----------------------------
# 11. Display final answer
# -----------------------------

print("\n========== ANSWER ==========\n")
print(answer)