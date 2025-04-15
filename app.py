import faiss
import pickle
import os
import numpy as np
import streamlit as st
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# Streamlit UI setup
st.set_page_config(page_title="Medical Chatbot", page_icon="💬")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Load FAISS index and text strings
with open('faiss_index.pkl', 'rb') as f:
    index, text_strings = pickle.load(f)

# Define a custom retriever
class CustomFaissRetriever(BaseRetriever):
    index: faiss.Index = None
    embeddings: object = None
    text_strings: list[str] = None
    k: int = 5

    def __init__(self, index: faiss.Index, embeddings: object, text_strings: list[str], k: int = 5):
        super().__init__()  # Call the base class's constructor
        self.index = index
        self.embeddings = embeddings
        self.text_strings = text_strings
        self.k = k

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Retrieve relevant documents based on the query."""
        query_vector = self.embeddings.embed_query(query)
        D, I = self.index.search(np.array([query_vector], dtype=np.float32), self.k)
        return [Document(page_content=self.text_strings[i]) for i in I[0]]

    def retrieve(self, query: str) -> list[Document]:
        """Override retrieve method to match LangChain's contract.""" 
        return self._get_relevant_documents(query)

retriever = CustomFaissRetriever(index=index, embeddings=embeddings, text_strings=text_strings)

# Prompt template setup
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Use Groq LLMs and Add a Model Selection dropdown
model_options = ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "gemma2-9b-it"]
selected_model = st.sidebar.selectbox("Choose a Model", model_options)

if selected_model == "llama-3.3-70b-versatile":
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

elif selected_model == "deepseek-r1-distill-llama-70b":
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="deepseek-r1-distill-llama-70b"
    )
elif selected_model == "gemma2-9b-it":
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="gemma2-9b-it"
    )
else:
    raise ValueError(f"Invalid model selected: {selected_model}")

# Initialize the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Streamlit UI
st.title("💬 End-to-End Medical Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Ask your question here...")

if user_query:
    st.session_state.chat_history.append(("user", user_query))
    with st.spinner("Generating response..."):
        result = qa({"query": user_query})
        bot_reply = result["result"]
    st.session_state.chat_history.append(("bot", bot_reply))

# Chat history display
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "user" else "assistant"):
        st.markdown(message)

# Add Features Section
st.sidebar.markdown("### Features:")
st.sidebar.markdown("- **Powered by Groq LLMs:**  For accurate and reliable medical information.")
st.sidebar.markdown("- **Medical Expertise:** Fine-tuned on a specialized dataset of health knowledge.")
st.sidebar.markdown("- **Fast & Accurate Answers:** Quickly finds the information you need.") 
