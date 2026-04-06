import streamlit as st
import tempfile
import hashlib
import torch
import os
import faiss

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder

#config
st.set_page_config(
    page_title = "Scholia AI",
    page_icon = "🥀",
    layout = "wide"
)
st.title("Scholia AI")
st.sidebar.title("Upload Document")
uploaded_files = st.sidebar.file_uploader(
    "Choose a PDF",
    type = "pdf",
    accept_multiple_files = False
)

#torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

#loaders(cache)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name = "BAAI/bge-small-en",
        model_kwargs = {"device": DEVICE}
    )

@st.cache_resource
def load_llm():
    return OllamaLLM(
        model = "llama3",
        num_ctx = 4096,
        temperature = 0.2
    )

@st.cache_resource
def load_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device = DEVICE)

#helpers
def format_context(docs):
    """Structured context for LLM"""
    context = ""
    sources = []
    
    for doc, score in docs:
        page = doc.metadata.get("page", 0) + 1
        context += f"[Page {page}]\n{doc.page_content}\n\n"
        sources.append(page)
    
    return context, sorted(set(sources))

def build_prompt(context, query):
    return f"""

You are strict document Q/A assistant.

RULES:
- Answer ONLY from context
- Maximum 4 lines
- Include exact quote from context with page number
- Do not explain beyond context
- If missing, then say "Not found in document"

Context:
{context}

Question:
{query}

Answer:
"""

def limit_context(docs, max_chars = 1500):
    context = ""
    sources = []

    for doc in docs:
        text = doc.page_content.strip()
        page = doc.metadata.get("page", 0) + 1

        if len(context) + len(text) > max_chars:
            break

        context += f"[Page {page}]\n{text}\n\n"
        sources.append(page)

    return context, sorted(set(sources))

#extract relevance
def extract_relevant_sentences(docs, query, max_sentences = 5):
    STOPWORDS = {
        "what", "is", "the", "a", "an", "of", "in", "on", "for",
        "to", "and", "or", "by", "with", "how"
    }

    query_words = set(
        word for word in query.lower().split()
        if word not in STOPWORDS
    )

    selected = []

    for doc in docs:
        sentences = doc.page_content.split(".")

        for s in sentences:
            s = s.strip()
            if len(s) < 20:
                continue

            sentence_words = set(s.lower().split())
            common_words = query_words.intersection(sentence_words)

            if len(common_words) >= 2:
                selected.append((s, doc.metadata.get("page", 0) + 1))

            if len(selected) >= max_sentences:
                break
        
        if len(selected) >= max_sentences:
            break
    
    context = ""
    sources = set()

    for sent, page in selected:
        context += f"[Page {page}] {sent}.\n"
        sources.add(page)
    
    return context, sorted(sources)

#query intent
def detect_intent(query):
    q = query.lower()
    if "summarize" in q:
        return "summary"
    elif "explain" in q:
        return "explain"
    elif "define" in q or "what is" in q:
        return "definition"
    return "general"

#history
def build_chat_context(messages, limit = 3):
    history = ""
    for msg in messages[-limit:]:
        role = msg["role"]
        content = msg["content"]
        history += f"{role.upper()}: {content}\n"
    return history

#main
if uploaded_files:
    file_bytes = uploaded_files.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    index_dir = f"faiss_index_{file_hash}"

    file_changed = (
        "processed_file" not in st.session_state or
        st.session_state.processed_file != file_hash
    )

    #save temp file
    if file_changed:
        with tempfile.NamedTemporaryFile(delete = False, suffix = "pdf") as tmp:
            tmp.write(file_bytes)
            st.session_state.pdf_path = tmp.name

    embeddings = load_embeddings()

    #process pdf, build db
    if file_changed:
        with st.spinner("Indexing Document..."):
            try:
                loader = PyPDFLoader(st.session_state.pdf_path)
                pages = loader.load()
            except Exception as e:
                st.error(f"Error loading pdf: {e}")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size = 250,
                chunk_overlap = 50
            )

            chunks = splitter.split_documents(pages)

            st.sidebar.write(f"Pages : {len(pages)}")
            st.sidebar.write(f"Chunks : {len(chunks)}")

            db = FAISS.from_documents(chunks, embeddings)

            os.makedirs(index_dir, exist_ok = True)
            db.save_local(index_dir)

            st.session_state.db = db
            st.session_state.processed_file = file_hash
            st.session_state.messages = []
    else:
        #load existing index
        if "db" not in st.session_state and os.path.exists(index_dir):
            st.session_state.db = FAISS.load_local(
                index_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )

    if DEVICE == "cuda":
        res = faiss.StandardGpuResources()
        db_index = db.index
        db.index = faiss.index_cpu_to_gpu(res, 0, db_index)

    st.sidebar.success("Document Ready")

    #load model
    llm = load_llm()

    #chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    #input
    query = st.chat_input("Ask Scholia")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        #refining query
        refined_query = llm.invoke(f"Rewrite this for document search:\n{query}")

        db = st.session_state.db

        #retrieval
        docs = db.max_marginal_relevance_search(
            refined_query,
            k = 4,
            fetch_k = 8
        )

        #reranking
        reranker = load_reranker()

        pairs = [(query, doc.page_content) for doc in docs]

        if all(len(doc.page_content) < 200 for doc in docs):
            scores = [1.0]*len(docs)
        else:
            scores = reranker.predict(pairs)

        reranked = sorted(
            zip(docs, scores),
            key = lambda x: x[1],
            reverse = True
        )

        #reranker optional if score low
        if max(scores) < 0.1:
            top_docs = docs[:3]
        
        else:
            THRESHOLD = max(scores) * 0.6
            filtered = [
                (doc, score) for (doc, score) in reranked
                if score > THRESHOLD
            ]
            top_docs = [doc for doc, score in filtered][:3]

        if not top_docs:
            response = "Not found in document"
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            st.stop()

        #filtered
        top_docs = [doc for doc, score in filtered][:3]

        if not top_docs:
            response = "No relevant context found in the document."
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.stop()

        #extract
        context, sources = limit_context(top_docs, max_chars = 3000)

        #fallback to raw top_doc
        if not context.strip():
            top_doc = top_docs[0]
            page = top_doc.metadata.get("page", 0) + 1
            context = f"[Page {page}]\n{top_doc.page_content}"
            sources = [page]

        #chat history + intent + prompt
        chat_history = build_chat_context(st.session_state.messages)
        intent = detect_intent(query)

        extra = {
            "summary": "Give a concise summary.",
            "explain": "Explain in simple terms.",
            "definition": "Give a clear definition."
        }.get(intent, "")

        prompt = f"""
        Conversation History:
        You are a strict document QNA assistant.
        Answer ONLY from the context below.
        If the answer is not in the context, say "Not found in document"
        
        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        #response
        with st.spinner("Thinking..."):
            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in llm.stream(prompt):
                    full_response += chunk
                    response_placeholder.write(full_response)
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.stop()
        
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        #copy
        with st.expander("Copy Latest Answer"):
            st.text_area("Answer", full_response, height = 150)

        #sources
        st.markdown("Sources")
        for doc in top_docs:
            page = doc.metadata.get("page", 0) + 1
            preview = doc.page_content[:200]

            st.markdown(f"**Page {page} Preview:**")
            st.write(preview + "...")

        for doc in docs:
            page = doc.metadata.get("page", 0) + 1
            st.markdown(f"**Page {page}**")
            st.write(doc.page_content)

else:
    st.info("Upload a PDF to start chat")