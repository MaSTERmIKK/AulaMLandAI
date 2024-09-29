import chromadb
from chromadb.config import Settings
import numpy as np

# Inizializziamo il client ChromaDB
chroma_client = chromadb.Client(Settings())

# Creiamo una nuova collezione denominata "immagini_embeddings"
collezione = chroma_client.create_collection(name="immagini_embeddings")

# Supponiamo di avere embeddings di immagini
embeddings = [
    np.random.rand(128),  # Embedding dell'immagine 1
    np.random.rand(128),  # Embedding dell'immagine 2
    np.random.rand(128)   # Embedding dell'immagine 3
]

# Convertiamo gli embeddings in liste per compatibilità
embeddings = [embedding.tolist() for embedding in embeddings]

# Aggiungiamo gli embeddings alla collezione
collezione.add(
    embeddings=embeddings,
    metadatas=[
        {"nome": "immagine1.jpg", "categoria": "natura"},
        {"nome": "immagine2.jpg", "categoria": "città"},
        {"nome": "immagine3.jpg", "categoria": "natura"}
    ],
    ids=["img1", "img2", "img3"]
)
