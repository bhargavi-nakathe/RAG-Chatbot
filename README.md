# RAG-Chatbot

Its a chatbot where it  help you by answering questions on Artificial Intelligence, Machine Learning, and Deep Learning by Oswald Campesato.
# RAG-Based Chatbot with Google Gemini API 🚀

This project is a Retrieval-Augmented Generation (RAG) chatbot that provides accurate and context-based responses by retrieving relevant knowledge from a local document database. Unlike standard LLMs that rely solely on their internal knowledge, this chatbot uses **ChromaDB** for semantic search and **Google Gemini 2.0 Flash** for response generation, ensuring responses are grounded in your data.

## 🔍 How It Works

- **Document Embedding**: Text data is embedded using `GoogleGenerativeAIEmbeddings`.
- **Storage & Retrieval**: ChromaDB stores the vectorized documents and allows efficient similarity-based retrieval.
- **Response Generation**: The chatbot uses `ChatGoogleGenerativeAI` to generate answers strictly based on the retrieved chunks.
- **Interface**: The entire system is wrapped in a sleek and interactive **Gradio UI** for easy testing and deployment.

## 💡 Features

- Real-time streaming responses
- Strict use of external knowledge (no hallucination)
- Environment variable protection using `.env`
- Clean, minimal interface for interaction

## 🛠 Tech Stack

- Python, LangChain
- Google Gemini API (`models/embedding-001`, `gemini-2.0-flash`)
- ChromaDB
- Gradio
- dotenv

## 🚀 Getting Started

1. Clone this repo
2. Create a `.env` file with your `GEMINI_API_KEY`
3. Run the script: `python app.py`
4. Start chatting!

### 📸 Demo

![Chatbot Screenshot]("D:\buildbot\chatbot.png")



Feel free to ask @nbhargavi204@gmail.com
