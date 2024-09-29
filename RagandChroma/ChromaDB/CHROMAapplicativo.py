import chromadb
from chromadb.config import Settings

# Inizializziamo il client ChromaDB con le impostazioni di default
chroma_client = chromadb.Client(Settings())

# Creiamo una nuova collezione denominata "miei_documenti"
collezione = chroma_client.create_collection(name="miei_documenti")

# Aggiungiamo documenti alla collezione
collezione.add(
    documents=[
        "Questo è il primo documento.",
        "Ecco il secondo documento.",
        "Il terzo documento è qui."
    ],
    metadatas=[
        {"categoria": "testo"},
        {"categoria": "testo"},
        {"categoria": "testo"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# Eseguiamo una query per trovare documenti simili
risultati = collezione.query(
    query_texts=["secondo documento"],
    n_results=2
)

print(risultati)
