from rag_pipeline import (
    charger_pdfs,
    decouper_documents,
    creer_vectorstore,
    creer_chatbot,
    poser_question
)

print("=== Test du pipeline RAG ===\n")

# 1. Charger les PDFs
print("1. Chargement des PDFs...")
documents = charger_pdfs("./pdfs")
print(f"   {len(documents)} pages chargées\n")

# 2. Découper en chunks
print("2. Découpage en chunks...")
chunks = decouper_documents(documents)
print(f"   {len(chunks)} chunks créés\n")

# 3. Créer le vectorstore
print("3. Création du vectorstore...")
vectorstore = creer_vectorstore(chunks)
print("   Vectorstore prêt\n")

# 4. Créer le chatbot
print("4. Initialisation du chatbot...")
chaine, retriever = creer_chatbot(vectorstore)
print("   Chatbot prêt\n")

# 5. Poser des questions
questions = [
    "De quoi parle ce document ?",
    "Quels sont les points principaux ?",
]

for question in questions:
    print(f"Question : {question}")
    reponse, sources = poser_question(chaine, retriever, question)
    print(f"Réponse  : {reponse[:300]}...")
    print(f"Sources  : {[doc.metadata.get('source', '?') for doc in sources[:2]]}")
    print()