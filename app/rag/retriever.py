import chromadb
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings

embeddings = OllamaEmbeddings(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url
)

client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
collection = client.get_or_create_collection("diabetes_knowledge")

def retrieve_context(query: str, top_k: int = 3) -> str:
    query_vector = embeddings.embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    
    if not results["documents"][0]:
        return "No relevant context found."
    
    return "\n\n".join(results["documents"][0])