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

if __name__ == "__main__":
    ingest()