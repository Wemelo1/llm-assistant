import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="JAMB/WAEC AI Tutor",
    page_icon="🎓",
    layout="wide"
)

# ============================================
# CUSTOM STYLING
# ============================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .subject-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        margin: 5px;
        font-weight: bold;
    }
    .score-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <h1 style="color: #00d4aa;">🎓 JAMB/WAEC AI Tutor</h1>
    <p style="color: #888;">Your Personal AI Study Assistant for Nigerian Students</p>
    <p style="color: #666; font-size: 12px;">Powered by LLaMA 3.3 + RAG Technology</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_..."
    )

    st.divider()
    st.header("📚 Select Subject")

    subject = st.selectbox(
        "Subject",
        ["All Subjects", "Mathematics", "Physics",
         "Chemistry", "English", "Biology"]
    )

    st.divider()
    st.header("🎯 Study Mode")

    mode = st.selectbox(
        "Mode",
        [
            "Ask Questions",
            "Practice Quiz",
            "Explain Concept",
            "Past Questions",
            "Quick Summary"
        ]
    )

    st.divider()

    # Score tracker
    if "score" not in st.session_state:
        st.session_state.score = {"correct": 0, "total": 0}

    st.header("📊 Your Score")
    correct = st.session_state.score["correct"]
    total = st.session_state.score["total"]

    if total > 0:
        percentage = int((correct/total) * 100)
        st.metric("Score", f"{correct}/{total}", f"{percentage}%")
        if percentage >= 70:
            st.success("Excellent! Keep it up! 🔥")
        elif percentage >= 50:
            st.warning("Good effort! Keep practicing! 💪")
        else:
            st.error("Keep studying! You'll get there! 📖")
    else:
        st.info("Start answering questions to track your score!")

    if st.button("Reset Score"):
        st.session_state.score = {"correct": 0, "total": 0}
        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built by Pr0_M1se 🚀")
    st.caption("LLM Engineer in Training")

# ============================================
# LOAD KNOWLEDGE BASE
# ============================================
@st.cache_resource
def load_knowledge_base():
    """Load and index the JAMB questions database"""
    questions_path = os.path.join(
        os.path.dirname(__file__), "questions.txt"
    )

    loader = TextLoader(questions_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

# Load knowledge base
with st.spinner("Loading knowledge base..."):
    try:
        vectorstore = load_knowledge_base()
        st.sidebar.success("Knowledge base loaded!")
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        st.stop()

# ============================================
# CHAT HISTORY
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": """Welcome to JAMB/WAEC AI Tutor! 🎓

I'm here to help you prepare for your exams. Here's what I can do:

- **Ask Questions** — Ask me anything about your subjects
- **Practice Quiz** — Type 'quiz me on [topic]' for practice questions
- **Explain Concept** — Type 'explain [concept]' for clear explanations
- **Past Questions** — Type 'past questions on [topic]' for exam questions

Select your subject and study mode from the sidebar, then let's begin! 💪

**What subject would you like to start with?**"""
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================
# MAIN CHAT LOGIC
# ============================================
if prompt := st.chat_input("Ask a question, request a quiz, or type a topic..."):

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
                    temperature=0.4
                )
                parser = StrOutputParser()

                # Search knowledge base
                search_query = prompt
                if subject != "All Subjects":
                    search_query = f"{subject}: {prompt}"

                relevant_docs = vectorstore.similarity_search(
                    search_query, k=4
                )
                context = "\n\n".join([
                    doc.page_content for doc in relevant_docs
                ])

                # Detect mode from prompt
                prompt_lower = prompt.lower()

                if "quiz" in prompt_lower or mode == "Practice Quiz":
                    template = PromptTemplate(
                        input_variables=["context", "subject", "topic"],
                        template="""You are a JAMB/WAEC exam coach for Nigerian students.

Using this context from the knowledge base:
{context}

Generate 5 multiple choice questions about {topic} for {subject}.
Format EXACTLY like this:

**Question 1:** [question]
A) [option]
B) [option]
C) [option]
D) [option]
**Answer:** [correct letter]
**Explanation:** [brief explanation]

---

**Question 2:** [question]
A) [option]
B) [option]
C) [option]
D) [option]
**Answer:** [correct letter]
**Explanation:** [brief explanation]

---

**Question 3:** [question]
A) [option]
B) [option]
C) [option]
D) [option]
**Answer:** [correct letter]
**Explanation:** [brief explanation]

---

---

**Question 4:** [question]
A) [option]
B) [option]
C) [option]
D) [option]
**Answer:** [correct letter]
**Explanation:** [brief explanation]

---

**Question 5:** [question]
A) [option]
B) [option]
C) [option]
D) [option]
**Answer:** [correct letter]
**Explanation:** [brief explanation]"""
                    )
                    topic = prompt.replace("quiz me on", "").replace(
                        "quiz", "").strip() or subject
                    chain = template | llm | parser
                    response = chain.invoke({
                        "context": context,
                        "subject": subject,
                        "topic": topic
                    })
                    st.session_state.score["total"] += 3

                elif "explain" in prompt_lower or mode == "Explain Concept":
                    template = PromptTemplate(
                        input_variables=["context", "concept", "subject"],
                        template="""You are an expert Nigerian exam tutor.

Knowledge base context:
{context}

Explain this concept for a Nigerian student preparing for JAMB/WAEC:
Concept: {concept}
Subject: {subject}

Structure your explanation as:
1. **Simple Definition** (1-2 sentences)
2. **Key Points** (bullet points)
3. **Nigerian Context Example** (relate to Nigeria where possible)
4. **Exam Tips** (what to remember for JAMB/WAEC)
5. **Practice Question** (one sample question)"""
                    )
                    concept = prompt.replace("explain", "").strip()
                    chain = template | llm | parser
                    response = chain.invoke({
                        "context": context,
                        "concept": concept,
                        "subject": subject
                    })

                elif "past question" in prompt_lower or mode == "Past Questions":
                    template = PromptTemplate(
                        input_variables=["context", "topic"],
                        template="""You are a JAMB past questions expert.

From this knowledge base:
{context}

Provide 3 past-exam style questions on: {topic}

For each question include:
- The question
- 4 options (A, B, C, D)
- The correct answer
- Year-style reference (e.g., "JAMB 2019 style")
- Brief explanation"""
                    )
                    topic = prompt.replace("past questions on", "").replace(
                        "past questions", "").strip()
                    chain = template | llm | parser
                    response = chain.invoke({
                        "context": context,
                        "topic": topic
                    })

                elif "summary" in prompt_lower or mode == "Quick Summary":
                    template = PromptTemplate(
                        input_variables=["context", "topic", "subject"],
                        template="""You are a JAMB/WAEC study guide writer.

Knowledge base:
{context}

Create a quick revision summary for {topic} in {subject}.

Format as:
## {topic} — Quick Revision

**Key Definitions:**
- [term]: [definition]

**Important Formulas/Rules:**
- [formula or rule]

**Must Remember for Exam:**
- [key point]

**Common Mistakes to Avoid:**
- [mistake]"""
                    )
                    topic = prompt.replace("summary", "").replace(
                        "summarize", "").strip() or subject
                    chain = template | llm | parser
                    response = chain.invoke({
                        "context": context,
                        "topic": topic,
                        "subject": subject
                    })

                else:
                    # Default Q&A mode
                    template = PromptTemplate(
                        input_variables=["context", "question", "subject"],
                        template="""You are a helpful JAMB/WAEC tutor for Nigerian students.

Knowledge base context:
{context}

Subject focus: {subject}

Answer this question clearly for a Nigerian student:
{question}

- Be concise but complete
- Use Nigerian context where relevant
- If it's a calculation, show the working
- End with one exam tip if relevant"""
                    )
                    chain = template | llm | parser
                    response = chain.invoke({
                        "context": context,
                        "question": prompt,
                        "subject": subject
                    })

                st.markdown(response)

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)

# ============================================
# FOOTER TIPS
# ============================================
st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("💡 **Tip:** Type 'quiz me on quadratic equations' for practice")
with col2:
    st.info("📖 **Tip:** Type 'explain osmosis' for clear explanations")
with col3:
    st.info("📝 **Tip:** Type 'past questions on Newton laws' for exam prep")
with col4:
    st.info("⚡ **Tip:** Type 'summary of organic chemistry' for revision")