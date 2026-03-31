from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

PROMPT_TEMPLATE = """
Tu es un assistant qui répond aux questions en te basant
UNIQUEMENT sur les documents fournis ci-dessous.
Si la réponse n'est pas dans les documents, dis-le clairement.
Ne fabrique jamais d'informations.

Contexte :
{context}

Question : {question}

Réponse :"""

def charger_pdfs(dossier):
    documents = []
    for fichier in os.listdir(dossier):
        if fichier.endswith('.pdf'):
            chemin = os.path.join(dossier, fichier)
            loader = PyMuPDFLoader(chemin)
            docs = loader.load()
            documents.extend(docs)
            print(f"Chargé : {fichier} ({len(docs)} pages)")
    return documents

def decouper_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Nombre de chunks : {len(chunks)}")
    return chunks

def creer_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"Vectorstore créé !")
    return vectorstore

def creer_chatbot(vectorstore):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=1000
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chaine = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chaine, retriever

def poser_question(chaine, retriever, question):
    reponse = chaine.invoke(question)
    sources = retriever.invoke(question)
    return reponse, sources