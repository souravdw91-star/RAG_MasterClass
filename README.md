# RAG MasterClass

A practical Retrieval-Augmented Generation (RAG) project for building question-answering systems over custom knowledge sources using LangChain, Hugging Face embeddings, Chroma, and a Groq-hosted LLM.

This repository contains both:

- interactive notebook-based learning examples
- Python script versions that mirror the same RAG workflow in a reusable package format

## Project Overview

The project demonstrates the standard RAG pipeline:

1. Load a custom document or text content
2. Split it into chunks
3. Generate embeddings for each chunk
4. Store the embeddings in a vector store
5. Retrieve the most relevant passages for a query
6. Pass the retrieved context to an LLM for answer generation

This is useful for learning how document-grounded AI systems work and for building a foundation for customer support, internal knowledge search, or document Q&A applications.

## Tech Stack

- Python 3.12+
- LangChain
- LangChain Community
- LangChain Groq
- LangChain Hugging Face Embeddings
- ChromaDB
- Jupyter Notebook
- Python-dotenv
- PyPDF (for PDF-based workflow)

## Repository Structure

```text
RAG_MasterClass/
├── 1.0_RAG_With_Own_Text.ipynb          # Notebook: custom text RAG example
├── Ex_1_0_Document.ipynb                # Notebook: PDF-based RAG example
├── pyproject.toml                       # Project metadata and dependencies
├── README.md                            # Project documentation
├── requirements.txt                      # Installed dependency snapshot
├── uv.lock                              # Lock file for uv-managed environment
├── .env                                 # Local environment variables (not committed)
├── .gitignore
├── src/
│   └── rag_masterclass/
│       ├── __init__.py                  # Package init
│       ├── rag_with_own_text.py         # Script version of custom text workflow
│       └── ex_1_0_document.py           # Script version of PDF workflow
├── .venv/                               # Local virtual environment
└── .ipynb_checkpoints/                  # Jupyter checkpoint files
```

## Prerequisites

Before running the project, make sure you have:

- Python 3.12 or newer
- A Groq API key
- A virtual environment manager such as `uv` or `venv`
- Jupyter Notebook installed if you want to run the notebooks directly

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd RAG_MasterClass
```

### 2. Create the environment

Using `uv`:

```bash
uv sync
```

Or using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
pip install -e .
```

### 3. Add your environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
```

> The model value can be adjusted based on the Groq model available in your account.

## Running the Notebooks

From the project root:

```bash
jupyter notebook
```

Then open either:

- `1.0_RAG_With_Own_Text.ipynb`
- `Ex_1_0_Document.ipynb`

These notebooks walk through the same logic as the Python scripts in the package.

## Running the Python Modules

You can run the equivalent script versions directly:

```bash
python -m rag_masterclass.rag_with_own_text
python -m rag_masterclass.ex_1_0_document
```

### Script behavior

- `rag_with_own_text.py`
  - creates a custom text document
  - splits the content into chunks
  - embeds the chunks using `all-MiniLM-L6-v2`
  - stores them in Chroma
  - retrieves the top matching chunks for a query
  - sends the retrieved context to Groq for answer generation

- `ex_1_0_document.py`
  - loads a PDF file named `pdf.pdf` from the project root
  - processes it the same way as the notebook
  - creates a vector store for the PDF content
  - runs a query against the retrieved context

## Example Workflow

The core pipeline is similar to this:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)

context = vectorstore.similarity_search("what is AI?", k=3)
```

Then the most relevant passages are passed to the Groq model as context to generate a grounded answer.

## Notes

- This project is intended for learning and experimentation.
- The vector store is created locally, which is ideal for demos and prototype systems.
- For production implementations, you would typically add persistence, improved chunking, metadata filters, memory, prompt strategies, and evaluation metrics.

## License

This project is for educational and learning purposes.

## Contact

For questions or improvements, you can reach out through the project repository or the author contact information configured in the project metadata.
