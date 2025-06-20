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
    model="models/embedding-001",
    google_api_key=api_key  
)

# Replace ChatOpenAI with ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=api_key
)

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 20
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot
def stream_response(message, history):

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # DEBUG: Print retrieved documents
    print("🔍 RETRIEVED CHUNKS:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Chunk {i} ---\n{doc.page_content}\n")

    # add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"


    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        You are a helpful AI assistant that answers questions using the provided information in the "Knowledge" section below.

        - If the answer is clearly present in the knowledge, respond using that.
        - If the answer is not found, try your best to answer using general knowledge.
        - If someone greets you with "hi" or "hello", reply: "Hello, I am a chatbot assistant."
        - If someone asks "In what ways can you assist me?", reply: "I can help you by answering questions on Artificial Intelligence, Machine Learning, and Deep Learning by Oswald Campesato."
        - If you're unsure about the answer, say: "I'm sorry, I couldn't find the answer based on the information I have."

        Question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        #print(rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch(share=True)