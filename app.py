import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

    if st.button("Clear Chat"):
        st.session_state.messages = []
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
