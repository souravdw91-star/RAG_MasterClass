import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


MY_TEXT = """Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in engineering, mathematics, and computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximise their chances of achieving defined goals.

High-profile applications of AI include advanced web search engines, chatbots, virtual assistants, autonomous vehicles, play and analysis in strategy games (e.g., chess and Go), and content generation (e.g., text, images, audio, and videos).

The traditional goals of AI research include learning, reasoning, knowledge representation, planning, natural language processing, and perception, as well as support for robotics. To reach these goals, AI researchers use techniques including state space search and mathematical optimisation, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics. AI also draws upon psychology, linguistics, philosophy, neuroscience, and other fields. Some companies, such as OpenAI, Google DeepMind, and Meta, aim to create artificial general intelligence (AGI)—AI that can complete nearly any cognitive task at least as well as a human.

Artificial intelligence was founded as an academic discipline in 1956. The field went through multiple cycles of optimism throughout its history, followed by periods of disappointment and loss of funding, known as AI winters. Funding and interest increased substantially after 2012, when graphics processing units (GPUs) started being used to accelerate neural networks, and deep learning outperformed previous AI techniques. This growth accelerated further after 2017 with the transformer architecture. The 2020s AI boom was spurred on largely by advances in generative AI, which can create and modify media. In addition to AI safety and unintended consequences and harms from the use of AI, ethical concerns, AI's long-term effects, environmental effects, and potential existential risks have prompted discussions of AI regulation.
"""


def load_llm() -> ChatGroq:
    """Initialize the Groq LLM from environment variables."""
    load_dotenv()

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        temperature=0.0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


def create_documents() -> list[Document]:
    """Wrap the custom text in a LangChain Document object."""
    return [
        Document(
            page_content=MY_TEXT,
            metadata={"source": "ABC", "documentID": "doc1"},
        )
    ]


def build_vectorstore(docs: list[Document]):
    """Split the document, create embeddings, and store them in Chroma."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    return chunks, embeddings, vectorstore


def run_rag_query(vectorstore, llm: ChatGroq, query: str = "what is AI?", k: int = 3):
    """Retrieve the top-k chunks and ask the model to answer from that context."""
    context = vectorstore.similarity_search(query, k=k)
    print("Retrieved context:\n")
    for item in context:
        print(item.page_content)
        print("\n---\n")

    response = llm.invoke(f"{query} you can answer using following context: {context}")
    print("\nLLM Response:\n")
    print(response.content)
    return response


def main() -> None:
    """Run the custom text RAG workflow from the notebook."""
    llm = load_llm()
    docs = create_documents()
    _, _, vectorstore = build_vectorstore(docs)
    run_rag_query(vectorstore, llm, query="what is AI?")


if __name__ == "__main__":
    main()
