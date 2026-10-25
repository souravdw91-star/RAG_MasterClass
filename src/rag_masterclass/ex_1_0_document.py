import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_llm() -> ChatGroq:
    """Initialize the Groq LLM from environment variables."""
    load_dotenv()

    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        temperature=0,
    )


def create_pdf_documents(pdf_path: str | os.PathLike[str]) -> list[Document]:
    """Load PDF content into LangChain Document objects."""
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({"source": "PDF", "documentID": "doc2"})

    return docs


def build_vectorstore(docs: list[Document]):
    """Split, embed, and store document chunks in Chroma."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    return chunks, embeddings, vectorstore


def run_rag_query(vectorstore, llm: ChatGroq, query: str = "what is chromadb?", k: int = 3):
    """Search relevant context and ask the LLM a grounded question."""
    context = vectorstore.similarity_search(query, k=k)
    combined_content = "\n\n".join(doc.page_content for doc in context)
    print("Retrieved context:\n")
    print(combined_content)

    response = llm.invoke(f"{query} you can answer using following context: {context}")
    print("\nLLM Response:\n")
    print(response.content)
    return response


def main() -> None:
    """Run the full PDF-based RAG workflow."""
    project_root = Path(__file__).resolve().parents[2]
    pdf_path = project_root / "pdf.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}. Place your PDF in the project root as 'pdf.pdf'.")

    llm = load_llm()
    docs = create_pdf_documents(pdf_path)
    _, _, vectorstore = build_vectorstore(docs)
    run_rag_query(vectorstore, llm)


if __name__ == "__main__":
    main()
