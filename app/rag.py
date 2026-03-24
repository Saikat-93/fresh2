from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_PATH = "data/faiss.index"
DOCS_PATH  = "data/documents.pkl"

documents = []
index = None


def _load():
    global index, documents
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
        print(f"[RAG] Loaded {len(documents)} chunks from disk.")


def _save():
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    print(f"[RAG] Saved {len(documents)} chunks to disk.")


def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def process_docs(text):
    global index, documents
    chunks = chunk_text(text)
    documents = chunks
    embeddings = EMBED_MODEL.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    _save()


def query_docs(query, k=3):
    global index, documents
    if index is None or len(documents) == 0:
        return ["No document uploaded yet."]
    q_emb = EMBED_MODEL.encode([query])
    D, I = index.search(np.array(q_emb, dtype=np.float32), k=k)
    return [documents[i] for i in I[0] if i < len(documents)]


# Load existing index on startup
_load()
