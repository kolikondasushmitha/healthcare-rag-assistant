# 🏥 Healthcare RAG Assistant

> An AI-powered healthcare chatbot that answers medical questions using 
> Retrieval Augmented Generation (RAG) — combining a local knowledge base 
> with a local LLM for accurate, context-aware responses.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Ollama](https://img.shields.io/badge/Ollama-TinyLlama-orange)

---

## 🧠 What is RAG?

**Retrieval Augmented Generation (RAG)** is an AI technique that:
1. **Retrieves** relevant information from a knowledge base
2. **Augments** the prompt with that context
3. **Generates** an accurate answer using an LLM

This means the AI answers based on **your data** — not just general training!

---

## ✨ Features

- 💬 Chat interface to ask health-related questions
- 🔍 Semantic search over medical knowledge base
- 🤖 Local LLM (no internet/API key needed)
- 📄 Easily expandable knowledge base
- ⚡ Fast responses with ChromaDB vector store
- 🔒 100% runs locally — your data stays private

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🦜 LangChain | RAG pipeline & LLM orchestration |
| 🗄️ ChromaDB | Vector database for semantic search |
| 🤖 Ollama + TinyLlama | Local LLM for answer generation |
| 🖥️ Streamlit | Chat UI interface |
| 🐍 Python 3.12 | Core language |

---

## 🏗️ Project Structure
---
healthcare-rag-assistant/
│
├── app/                    # Core application modules
│   ├── api/                # API routes
│   ├── core/               # Config and settings
│   ├── db/                 # Database handlers
│   ├── ml/                 # ML predictor
│   ├── rag/                # RAG retriever & generator
│   └── schemas/            # Data schemas
│
├── data/                   # Knowledge base & datasets
│   ├── diabetes_knowledge.txt
│   ├── diabetes.csv
│   └── confusion_matrix.png
│
├── scripts/                # Utility scripts
│   ├── ingest.py           # Load & store knowledge base
│   ├── query.py            # Terminal query interface
│   └── train_model.py      # ML model training
│
├── chroma_store/           # Vector DB storage (auto-generated)
├── app.py                  # Streamlit chat application
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables

## 🚀 How to Run

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai) installed

### Step 1 — Clone the repository
```bash
git clone https://github.com/kolikondasushmitha/healthcare-rag-assistant.git
cd healthcare-rag-assistant
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # Mac/Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Pull Ollama model
```bash
ollama pull tinyllama
```

### Step 5 — Ingest knowledge base
```bash
python scripts/ingest.py
```

### Step 6 — Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 💬 Example Questions

- *"What are the symptoms of diabetes?"*
- *"How can I prevent type 2 diabetes?"*
- *"What foods should a diabetic avoid?"*
- *"What is insulin resistance?"*

---

## 🔮 Future Improvements

- [ ] Add more diseases to knowledge base
- [ ] Upgrade to better LLM (Mistral, LLaMA2)
- [ ] Add PDF upload feature
- [ ] Deploy to cloud (Streamlit Cloud / Hugging Face)
- [ ] Add multilingual support

---

## 👩‍💻 Author

**Sushmitha Kolikonda**  
[![GitHub](https://img.shields.io/badge/GitHub-kolikondasushmitha-black)](https://github.com/kolikondasushmitha)

---

## 📄 License

This project is licensed under the MIT License.