import json
import os

import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI()


# --------------------------------------------------
# Enable CORS
# Allows our frontend to communicate with FastAPI
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load FAISS index
# --------------------------------------------------

index = faiss.read_index(
    "data/faiss/tesla.index"
)


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

with open(
    "data/faiss/metadata.json",
    "r"
) as f:

    metadata = json.load(f)


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Connect to Groq
# --------------------------------------------------

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# --------------------------------------------------
# Request format
# --------------------------------------------------

class QuestionRequest(BaseModel):

    question: str


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Tesla RAG backend is running!"
    }


# --------------------------------------------------
# RAG /ask endpoint
# --------------------------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    # ----------------------------------------------
    # Get user's question
    # ----------------------------------------------

    query = request.question


    # ----------------------------------------------
    # Convert question into embedding
    # ----------------------------------------------

    query_embedding = model.encode(
        [query]
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    faiss.normalize_L2(
        query_embedding
    )


    # ----------------------------------------------
    # Search FAISS
    # ----------------------------------------------

    k = 5

    scores, indices = index.search(
        query_embedding,
        k
    )


    # ----------------------------------------------
    # Build context for the LLM
    # ----------------------------------------------

    context = ""

    sources = []


    for rank, idx in enumerate(
        indices[0],
        start=1
    ):

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


        sources.append({

            "year": item["year"],

            "page": item["page"],

            "chunk_id": item["chunk_id"],

            "similarity": float(
                scores[0][rank - 1]
            )

        })


    # ----------------------------------------------
    # Create prompt
    # ----------------------------------------------

    prompt = f"""
You are a Tesla company research assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the context does not contain enough information to
answer the question, say that the available Tesla
documents do not contain enough information.

Do not make up facts.

Write your answer in plain text only. Do not use markdown
tables, bold asterisks, headers, or bullet symbols like "•".
If you need to list multiple items, use plain numbered
sentences instead (e.g. "1. ...", "2. ...").

User question:

{query}


Context:

{context}
"""


    # ----------------------------------------------
    # Send context + question to Groq
    # ----------------------------------------------

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    )


    # ----------------------------------------------
    # Get LLM answer
    # ----------------------------------------------

    answer = response.choices[0].message.content


    # ----------------------------------------------
    # Return result to frontend
    # ----------------------------------------------

    return {

        "question": query,

        "answer": answer,

        "sources": sources

    }