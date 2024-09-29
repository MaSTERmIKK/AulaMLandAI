import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Imposta la chiave API di OpenAI
os.environ["OPENAI_API_KEY"] = "LA_TUA_CHIAVE_API"

# Documento di esempio
documento = """
Python è un linguaggio di programmazione ad alto livello, interpretato e orientato agli oggetti.
La machine learning è un campo dell'intelligenza artificiale che si occupa della creazione di sistemi che apprendono automaticamente.
Le reti neurali sono modelli computazionali ispirati alla struttura del cervello umano.
L'apprendimento profondo è una sottoclasse della machine learning che utilizza reti neurali profonde.
Il linguaggio naturale è complesso e richiede tecniche avanzate per la sua elaborazione.
"""

# Suddividi il documento in chunk
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_text(documento)

# Crea gli embeddings e l'archivio vettoriale
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)

# Configura il modello di linguaggio
llm = OpenAI(temperature=0)

# Crea la catena RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Esegui una query di esempio
query = "Che cos'è l'apprendimento profondo?"
risposta = qa.run(query)

print("Domanda:", query)
print("Risposta:", risposta)
