# type: ignore
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
import gradio as gr

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Replace OpenAI embeddings with Google embeddings
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-04",
    google_api_key=api_key  
)

# Replace ChatOpenAI with ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,   # Slightly increased temperature to let the AI be more fluid and expressive
    google_api_key=api_key
)

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# --- CHANGED THIS VALUE ---
# 6 to 8 chunks of 1000 characters provides the perfect amount of context
num_results = 7
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot
def stream_response(message, history):

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content + "\n\n"

    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        # --- REWRITTEN PROMPT ---
        # Removed the restrictive "I'm sorry" catch-all so the AI can use general knowledge smoothly.
        rag_prompt = f"""
        You are an expert AI companion specializing in Artificial Intelligence, Machine Learning, and Deep Learning based on the textbook context provided below.

        INSTRUCTIONS:
        1. Use the "Reference Knowledge" section below as your primary source to answer the user's question.
        2. Do not just copy and paste text verbatim. Explain, rephrase, and elaborate on the concepts naturally.
        3. If the specific details aren't fully covered in the Reference Knowledge, seamlessly use your general AI/ML training knowledge to fill in the gaps and provide a complete, smart answer.
        4. Standard greetings like "hi" or "hello" should always be answered with: "Hello, I am a chatbot assistant."
        5. If asked "In what ways can you assist me?", always reply: "I can help you by answering questions on Artificial Intelligence, Machine Learning, and Deep Learning by Oswald Campesato."

        Conversation history: 
        {history}

        Reference Knowledge from the document: 
        {knowledge}

        Question: {message}
        Answer:
        """

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# initiate the Gradio app
chatbot = gr.ChatInterface(
    stream_response, 
    textbox=gr.Textbox(placeholder="Ask me anything about AI, ML, or Deep Learning...", container=False, autoscroll=True, scale=7)
)

if __name__ == "__main__":
    chatbot.launch(share=True)