from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load existing ChromaDB
embeddings = OllamaEmbeddings(model="tinyllama")
db = Chroma(persist_directory="./chroma_store", embedding_function=embeddings)

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Load LLM
llm = OllamaLLM(model="tinyllama")

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context: {context}

Question: {question}
""")

# Helper to format retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask a question
import sys
sys.path.append(".")

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings

def ingest():
    print("📂 Loading knowledge base...")
    with open("data/diabetes_knowledge.txt", "r") as f:
        text = f.read()
    print(f"   Loaded {len(text)} characters")

    print("\n✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.create_documents([text])
    print(f"   Created {len(chunks)} chunks")

    print("\n🔗 Loading Ollama embeddings...")
    print("   (make sure Ollama is running!)")
    embeddings = OllamaEmbeddings(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url
    )

    print("\n💾 Storing into ChromaDB...")
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    
    # Delete existing collection to avoid duplicates
    try:
        client.delete_collection("diabetes_knowledge")
        print("Cleared existing collection")
    except:
        pass
    
    collection = client.get_or_create_collection("diabetes_knowledge")

    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk.page_content)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[vector],
            documents=[chunk.page_content]
        )
        print(f"   ✅ Chunk {i+1}/{len(chunks)} stored")

    print(f"\n🎉 Done! {len(chunks)} chunks stored in ChromaDB")
    print(f"   Location: {settings.chroma_persist_dir}")
# Add this at the bottom of query.py instead of hardcoded question
while True:
    question = input("\n🏥 Ask a health question (or 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    result = rag_chain.invoke(question)
    print(f"\n💊 Answer: {result}")
if __name__ == "__main__":
    ingest()