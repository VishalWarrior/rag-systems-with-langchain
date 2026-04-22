from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. LLM
llm = ChatOllama(model="llama3")

# 2. Load data
loader = TextLoader("data.txt")
documents = loader.load()

# 3. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
docs = splitter.split_documents(documents)

# 4. Embeddings
embeddings = OllamaEmbeddings(model="llama3")

# 5. Vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# 6. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 7. Prompt
prompt = ChatPromptTemplate.from_template("""
Answer ONLY from the context.
If not found, say "I don't know".

Context:
{context}

Question:
{question}
""")

# 8. Format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 9. Runnable chain (LATEST)
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Run
while True:
    query = input("You: ")
    print("AI:", rag_chain.invoke(query))