import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import time
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Pinecone configuration
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)

index_name = 'product-catalog-index'
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# Connect to the index
myindex = pc.Index(index_name)
time.sleep(1)

# Set the Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize embeddings and vector store
embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = PineconeVectorStore(
    index=myindex,
    embedding=embed_model,
    text_key='Description'
)

# Initialize the chat history and system message
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

system_message = (
    "If a query lacks a direct answer e.g. durability, generate a response based on related features. "
    "You are a helpful and respectful shop assistant who answers queries relevant only to the shop. "
    "Please answer all questions politely. Use a conversational tone, like you're chatting with someone, "
    "not like you're writing an email. If the user asks about anything outside of the shop data like if they ask "
    "something irrelevant, simply say, 'I can only provide answers related to the shop, sir."
)

# Function to generate a response from Google Gemini
def generate_answer(system_message, chat_history, prompt):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')

    # Append the new prompt to the chat history
    chat_history.append(f"User: {prompt}")

    # Combine the system message with the chat history
    full_prompt = f"{system_message}\n\n" + "\n".join(chat_history) + "\nAssistant:"

    # Generate the response and add it to the chat history
    response = model.generate_content(full_prompt).text
    chat_history.append(f"Assistant: {response}")
    
    return response

# Function to create the relevant passage from vectorstore
def get_relevant_passage(query, vectorstore):
    results = vectorstore.similarity_search(query, k=1)
    if results:
        metadata = results[0].metadata
        context = (
            f"Product Name: {metadata.get('ProductName', 'Not Available')}\n"
            f"Brand: {metadata.get('ProductBrand', 'Not Available')}\n"
            f"Price: {metadata.get('Price', 'Not Available')}\n"
            f"Color: {metadata.get('PrimaryColor', 'Not Available')}\n"
            f"Description: {results[0].page_content}"
        )
        return context
    return "No relevant results found"

# Function to create the prompt for the chatbot
def make_rag_prompt(query, context):
    return f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"

# Streamlit interface
st.title("Shop Assistant Chatbot")

query = st.text_input("Ask your question:")

if st.button("Get Answer"):
    if query:
        # Retrieve relevant passage and create a prompt
        relevant_text = get_relevant_passage(query, vectorstore)
        prompt = make_rag_prompt(query, relevant_text)

        # Generate and display the final answer
        answer = generate_answer(system_message, st.session_state.chat_history, prompt)
        st.write("Answer:", answer)

        # Display the chat history
        with st.expander("Chat History"):
            for chat in st.session_state.chat_history:
                st.write(chat)
