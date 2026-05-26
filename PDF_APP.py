import os
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# PAGE CONFIG
st.set_page_config(
    page_title="Pr0_M1se's AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# HEADER
st.title("🤖 Pr0_M1se's AI Assistant")
st.caption("Powered by LLaMA 3.3 + LangChain")
st.divider()

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_..."
    )

    temperature = st.slider(
        "Creativity (Temperature)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1
    )

    persona = st.selectbox(
        "AI Persona",
        [
            "Helpful Assistant",
            "UNIBEN Engineering Tutor",
            "Personal Finanncial Advisor",
            "Strict Exam Coach",
            "Creative Writing helper"
        ]
    )

    st.divider()
    
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                    
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(docs)
                    
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2", 
                        encode_kwargs={"normalize_embeddings": True}
                    )
                    
                    st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
                    st.session_state.processed_file = uploaded_file.name
                    st.success("PDF Ready!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    os.remove(tmp_path)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        if "processed_file" in st.session_state:
            del st.session_state.processed_file
        st.rerun()

    st.divider()
    st.caption("Built by Pr0_M1se — LLM Engineer in training 🚀")

# PERSONAS
personas = {
    "Helpful Assistant": "You are a helpful, friendly AI assistant.",
    "UNIBEN Engineering Tutor": """You are an expert engineering tutor 
    at the University of Benin. You explain concepts clearly using 
    Nigerian examples and always encourage students.""",
    "Personal Financial Advisor": """You are an expert Personal financial 
    advisor helping undergraduates build wealth. You give practical 
    advice in Nigerian context using Naira.""",
    "Strict Exam Coach": """You are a strict but fair exam coach. 
    You quiz students, give tough love feedback, and push them 
    to perform at their best.""",
    "Creative Writing Helper": """You are a creative writing assistant 
    who helps craft engaging stories, poems,Engineering report and content."""
}

# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# CHAT INPUT
if prompt := st.chat_input("Ask me anything..."):

    if not api_key:
        st.error("Please enter your Groq API key in the sidebar!")
        st.stop()

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                llm = ChatGroq(
                    api_key=api_key,
                    model="llama-3.3-70b-versatile",
                    temperature=temperature
                )
                parser = StrOutputParser()

                system_prompt = personas[persona]
                
                context = ""
                if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
                    docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                    system_prompt += f"\n\nYou have access to the following extracted PDF context to help answer the user's question. ONLY use this context if relevant:\n{context}"

                # Build history string
                history = ""
                for msg in st.session_state.messages[:-1]:
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    history += f"{role}: {msg['content']}\n"

                template = (
                    "{system}\n\n"
                    "Conversation history:\n{history}\n\n"
                    "Human: {question}\n\n"
                    "Assistant:"
                )

                chat_template = PromptTemplate(
                    input_variables=["system", "history", "question"],
                    template=template
                )

                chain = chat_template | llm | parser

                response = chain.invoke({
                    "system": system_prompt,
                    "history": history,
                    "question": prompt
                })

                st.markdown(response)

                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
