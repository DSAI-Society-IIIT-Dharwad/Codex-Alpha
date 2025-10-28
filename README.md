# ⚖️ Legal Chat Assistant

An intelligent **Legal Chat Assistant** powered by OpenAI.  
This application provides users with a conversational interface to get assistance with their legal queries.

The backend is built with **Python (FastAPI, LangChain, LangGraph)**, and the frontend is a modern, responsive UI built with **React (Vite)** and styled with vanilla **CSS**.

---

## ✨ Features

- **Conversational AI:** Ask legal questions in natural language.
- **Fast & Scalable Backend:** Powered by FastAPI and Uvicorn.
- **Modern UI:** Clean and intuitive React interface.
- **Quick Setup:** Run locally in just a few steps.
- **RAG Agent:** Retrieval-Augmented Generation (RAG) Agent for more accurate answers to queries

---

## 🛠️ Tech Stack

### Backend
- **Language:** Python 3.9+
- **Framework:** LangChain, LangGraph, FastAPI
- **API:** OpenAI API
- **Server:** Uvicorn
- **Local URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Frontend
- **Library:** React (with JSX)
- **Build Tool:** Vite
- **Styling:** Vanilla CSS
- **Local URL:** [http://localhost:5173](http://localhost:5173)

---

## 🚀 Getting Started

1. Clone repository or download zip.

2. Generate API key from OpenAI dashboard

3. In folder ```backend``` create a ```.env``` file and add the ```OPENAI_API_KEY```

4. Open terminal and write the following commands:
    ```bash 
    cd backend
    pip install -r requirements.txt
    uvicorn app:app --reload
    ```
    ```bash
    cd frontend 
    npm install
    npm run dev
    ```

5. Run [https://localhost:5173/](http://127.0.0.1:8000) and you are good to go.


## 🔜 Upcoming features

1. File reading ability for case explanation (OCR capabilities).

2. New Chat button and Recent Chats feature

3. Multi-language support

4. Legal Form Generator: Auto-generate drafts for affidavits, agreements, or notices.
