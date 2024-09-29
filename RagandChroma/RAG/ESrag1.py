import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Imposta la chiave API di OpenAI
os.environ["OPENAI_API_KEY"] = "LA_TUA_CHIAVE_API"

# Lista di documenti (esempio su animali)
documenti = [
    "Il leone è un grande felino carnivoro originario dell'Africa.",
    "Gli elefanti sono i più grandi animali terrestri e hanno una proboscide.",
    "I delfini sono mammiferi marini noti per la loro intelligenza.",
    "Le aquile sono uccelli rapaci con una vista eccezionale.",
    "I panda giganti sono orsi originari della Cina e si nutrono principalmente di bambù."
]

# Suddividi i documenti in chunk
text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=0)
texts = text_splitter.split_text(" ".join(documenti))

# Crea gli embeddings e l'archivio vettoriale
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)

# Configura il modello di linguaggio
llm = OpenAI(temperature=0)

# Crea la catena RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Esegui alcune query di esempio
queries = [
    "Qual è il più grande animale terrestre?",
    "Quale animale ha una proboscide?",
    "Quale mammifero marino è noto per la sua intelligenza?",
    "Di cosa si nutrono principalmente i panda giganti?"
]

for query in queries:
    risposta = qa.run(query)
    print("Domanda:", query)
    print("Risposta:", risposta)
    print("---")
