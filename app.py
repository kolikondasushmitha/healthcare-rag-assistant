import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page config
st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Healthcare Assistant")
st.caption("Ask me anything about health and medical conditions!")

# Load RAG chain once (cached)
@st.cache_resource
def load_rag_chain():
    embeddings = OllamaEmbeddings(model="tinyllama")
    db = Chroma(persist_directory="./chroma_store", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="tinyllama")

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful healthcare assistant.
    Use ONLY the context below to answer the question.
    If the answer is not in the context, say "I don't have enough information to answer that."
    Give clear, simple and accurate answers. Do NOT make up information.

    Context: {context}

    Question: {question}

    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Load chain
with st.spinner("Loading healthcare assistant..."):
    rag_chain = load_rag_chain()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Ask a health question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(question)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})