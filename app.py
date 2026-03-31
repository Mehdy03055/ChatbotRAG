import streamlit as st
import os
from rag_pipeline import (
    charger_pdfs,
    decouper_documents,
    creer_vectorstore,
    creer_chatbot,
    poser_question
)

st.set_page_config(page_title="Chatbot PDF", layout="wide")
st.title("Chatbot sur tes PDFs")

# ---- Sidebar ----
with st.sidebar:
    st.header("Configuration")
    
    pdfs = st.file_uploader(
        "Uploader des PDFs",
        type="pdf",
        accept_multiple_files=True
    )
    
    if pdfs and st.button("Indexer les PDFs"):
        os.makedirs("./pdfs", exist_ok=True)
        for pdf in pdfs:
            with open(f"./pdfs/{pdf.name}", "wb") as f:
                f.write(pdf.read())
        
        with st.spinner("Indexation en cours..."):
            docs = charger_pdfs("./pdfs")
            chunks = decouper_documents(docs)
            vectorstore = creer_vectorstore(chunks)
            chaine, retriever = creer_chatbot(vectorstore)
            st.session_state.chaine = chaine
            st.session_state.retriever = retriever
        st.success(f"{len(pdfs)} PDF(s) indexé(s) !")

# ---- Historique ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources utilisées"):
                for src in msg["sources"]:
                    st.caption(f"Page {src.get('page', '?')} — {src.get('source', '?')}")

# ---- Zone de saisie ----
if question := st.chat_input("Pose une question sur tes documents..."):
    if "chaine" not in st.session_state:
        st.error("Upload et indexe des PDFs d'abord !")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans tes documents..."):
                reponse, sources = poser_question(
                    st.session_state.chaine,
                    st.session_state.retriever,
                    question
                )
            st.write(reponse)
            with st.expander("Sources utilisées"):
                for doc in sources:
                    st.caption(f"Page {doc.metadata.get('page', '?')} — {doc.metadata.get('source', '?')}")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": reponse,
            "sources": [doc.metadata for doc in sources]
        })

        