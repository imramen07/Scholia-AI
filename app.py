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
from collections import defaultdict

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
    accept_multiple_files = True
)

#torch
torch.set_num_threads(4)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: {DEVICE}")

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
        temperature = 0.0
    )

@st.cache_resource
def load_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device = DEVICE)

#helpers
def deduplicate_docs(docs):
    seen = set()
    unique_docs = []

    for doc in docs:
        text = doc.page_content.strip()
        if text not in seen:
            seen.add(text)
            unique_docs.append(doc)
    return unique_docs

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
                selected.append((
                    s,
                    doc.metadata.get("page", 0) + 1,
                    doc.metadata.get("source", "Unknown")
                ))

            if len(selected) >= max_sentences:
                break
        
        if len(selected) >= max_sentences:
            break
    
    context = ""
    pages_used = set()

    for sent, page, source in selected:
        context += f"[{source} - Page {page}] {sent}.\n"
        pages_used.add((source, page))
    
    return context, sorted(pages_used, key = lambda x: (x[0], x[1]))

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

#build prompt
def build_prompt(context, query, extra = "", history = ""):
    return f"""
You are strict document QNA assistant.

STRICT RULES:
- Answer ONLY from the provided context.
- DO NOT use outside knowledge.
- If not found, then say "Not found in document".
- Answer must be concise (Max 3 lines).
- Do NOT assume anything.
- Include source and page numbers like [file.pdf - Page X]

{extra}

Conversation History:
{history}

Context:
{context}

Question:
{query}

Answer:
"""

#main
if uploaded_files:
    file_data = []

    for f in uploaded_files:
        data = f.read()
        file_data.append((f.name, data))

    all_bytes = b"".join([data for _, data in file_data])
    file_hash = hashlib.md5(all_bytes).hexdigest()
    
    for f in uploaded_files:
        f.seek(0)

    index_dir = f"faiss_index_{file_hash}"

    file_changed = (
        "processed_file" not in st.session_state or
        st.session_state.processed_file != file_hash
    )

    embeddings = load_embeddings()

    #process pdf, build db
    if file_changed:
        with st.spinner("Indexing Document..."):
            try:
                all_pages = []
                
                for name, data in file_data:
                    with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp:
                        tmp.write(data)
                        tmp_path = tmp.name
                
                    try:
                        loader = PyPDFLoader(tmp_path)
                        pages = loader.load()

                    finally:
                        #delete tempfile
                        os.remove(tmp_path)

                    #attaching sources
                    for p in pages:
                        p.metadata["source"] = name

                    all_pages.extend(pages)

            except Exception as e:
                st.error(f"Error loading pdf: {e}")
                st.stop()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size = 300,
                chunk_overlap = 80
            )

            chunks = splitter.split_documents(all_pages)

            st.sidebar.write(f"Total pages: {len(all_pages)}")
            st.sidebar.write(f"Chunks: {len(chunks)}")

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
                allow_dangerous_deserialization = True
            )

    if DEVICE == "cuda" and "db" in st.session_state:
        try:
            res = faiss.StandardGpuResources()
            db = st.session_state.db
            db.index = faiss.index_cpu_to_gpu(res, 0, db.index)
            st.session_state.db = db
        except:
            pass

    st.sidebar.success("Document Ready")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with 💗 by Ramen")

    #load model
    llm = load_llm()

    #reranker
    reranker = load_reranker()

    #chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    #input
    query = st.chat_input("Ask Scholia")

    if query and not query.strip():
        st.stop()

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        #refining query
        refined_query = query.lower().strip()

        db = st.session_state.db

        #retrieval
        docs = db.max_marginal_relevance_search(
            refined_query,
            k = 5,
            fetch_k = 10
        )

        #if no docs retrieved
        if not docs:
            response = "Not found in document"
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            st.stop()

        pairs = [(query, doc.page_content) for doc in docs]

        #scoring
        if all(len(doc.page_content) < 200 for doc in docs):
            scores = [1.0] * len(docs)
        else:
            scores = reranker.predict(pairs)

        #sort by score
        reranked = sorted(
            zip(docs, scores),
            key = lambda x: x[1],
            reverse = True
        )

        #select
        if not scores:
            top_docs = docs[:3]

        elif max(scores) < 0.2:
            top_docs = docs[:3]

        else:
            THRESHOLD = max(0.2, max(scores) * 0.5)
            top_docs = [
                doc for doc, score in reranked if score >= THRESHOLD
            ][:3]

        if not top_docs:
            top_docs = [doc for doc, _ in reranked[:2]]

        #filtered
        top_docs = deduplicate_docs(top_docs)

        if not top_docs:
            response = "No relevant context found in the document."
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.stop()

        #extract
        context, pages_used = extract_relevant_sentences(top_docs, query)

        #fallback to raw top_doc
        if len(context.strip()) < 50:
            top_doc = top_docs[0]
            page = top_doc.metadata.get("page", 0) + 1
            source = top_doc.metadata.get("source", "Unknown")
            context = f"[{source} - Page {page}]\n{top_doc.page_content}"
            pages_used = [(source, page)]

        #chat history + intent + prompt
        chat_history = build_chat_context(st.session_state.messages)
        intent = detect_intent(query)

        extra = {
            "summary": "Give a concise summary.",
            "explain": "Explain in simple terms.",
            "definition": "Give a clear definition."
        }.get(intent, "")

        #call prompt
        prompt = build_prompt(context, query, extra, chat_history)

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

        best_doc = top_docs[0]
        source = best_doc.metadata.get("source", "Unknown")
        best_page = best_doc.metadata.get("page", 0) + 1

        st.markdown("Most Relevant")
        st.write(f"{source} - Page {best_page}")
        st.write(best_doc.page_content[:300] + "...")

        #sources
        st.markdown("Sources Used")
        grouped = defaultdict(list)
        for src, pg in pages_used:
            grouped[src].append(pg)

        for src in grouped:
            pages = sorted(grouped[src])
            st.write(f"{src}: Pages {', '.join(map(str, pages))}")

else:
    st.info("Upload a PDF to start chat")