# 💬 End-to-End Medical Chatbot

An intelligent, real-time AI chatbot built to deliver accurate, fast, and context-aware answers to your medical questions. Powered by Groq's lightning-fast LLMs, FAISS vector database, and Hugging Face embeddings, this chatbot is designed to make navigating through medical data easy, interactive, and insightful.

---

## 🌐 Live View

Check out the live demo of this End-to-End Medical chatbot: [Live Demo](https://end-to-end-medical-chatbot.streamlit.app/).

---

## 🚀 Features

- Powered by Groq LLMs: Choose from top-tier models like llama-3.3-70b-versatile, deepseek-r1-distill-llama-70b, and gemma2-9b-it for reliable and intelligent responses.

- Medical Expertise: The chatbot is trained and fine-tuned on health-focused dataset (PDF) to provide expert-level information.

- Real-Time Responses: Ask your medical questions and get immediate, contextual answers powered by Retrieval-Augmented Generation (RAG).

- Custom Model Selection: Flexibility to switch between Groq models as per your performance or accuracy needs.

- Chat History: Keeps track of your previous conversations for seamless user experience.

- Interactive Streamlit Interface: A clean, responsive UI for hassle-free interaction.

---

## 🧠 How It Works

- Data Extraction: PDF files are loaded using PyPDFLoader, then split into manageable text chunks using RecursiveCharacterTextSplitter.

- Embedding Generation: Chunks are embedded using Hugging Face's all-MiniLM-L6-v2 model.

- Vector Indexing: Chunks are stored in a FAISS vector index for fast similarity search.

- Query Retrieval: On user query, relevant document chunks are retrieved using a custom FAISS retriever.

- Answer Generation: Groq LLMs process the context and generate a concise, helpful response.

---

## 🛠️ Tech Stack

- Python – Backend logic and orchestration

- Streamlit – Interactive user interface

- FAISS – Fast vector search for document retrieval

- LangChain – Chain of tools and logic for embeddings, LLMs, and prompts

- Groq LLMs – High-performance large language models

- Hugging Face Embeddings – Text-to-vector conversion

- dotenv – Secure environment variable handling

---

## 📁 Directory Structure

```
.
├── app.py                   # Main Streamlit app
├── faiss_index.pkl          # Pickled FAISS index and text data
├── .env                     # Groq API key
├── src/
│   ├── helper.py            # Embeddings and data loading logic
│   └── prompt.py            # Prompt template for QA chain
```

---

## 📝 Prompt Template

```
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
```

---

## 🧪 Sample Usage

- Upload medical PDF documents to the specified folder.

- Launch the app with ``` streamlit run app.py ```

- Select a model from the sidebar.

- Ask your question and view accurate, context-aware answers instantly.

---

## 📦 Installation

```
pip install -r requirements.txt
```

Set up your ```.env``` file with:
```
GROQ_API_KEY=your_api_key_here
```

---

## 🧠 Example Embedding Code Snippet

```
from langchain_huggingface import HuggingFaceEmbeddings

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
```

---

## 🧲 FAISS Retriever Example

```
class CustomFaissRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        query_vector = self.embeddings.embed_query(query)
        D, I = self.index.search(np.array([query_vector], dtype=np.float32), self.k)
        return [Document(page_content=self.text_strings[i]) for i in I[0]]
```

---

## 📣 Final Thoughts

The End-to-End Medical Chatbot empowers users with instant access to trusted medical information. Whether you're analyzing health reports or simply curious about medical topics, this AI chatbot delivers speed, accuracy, and context – all in a user-friendly experience.

---

## 🖥️ App UI

![pic1](https://github.com/shibbir-ahmad24/End-to-End-Medical-Chatbot/blob/main/Figures/pic1.png)

![pic2](https://github.com/shibbir-ahmad24/End-to-End-Medical-Chatbot/blob/main/Figures/pic2.png)




