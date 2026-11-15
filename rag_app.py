
import os
import re
import unicodedata
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


# Load API keys from .env
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key is missing in .env file")

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault(
        "LANGSMITH_PROJECT",
        "South_Africa_RAG_Project"
    )
else:
    print("LangSmith API key not found. Tracing is disabled.")


# File and database paths
PDF_PATH = "South-Africa-Chapter-Manual-Internet-In-A-Box-Setup-for-Everyone.pdf"
FAISS_PATH = "faiss_index"

# True = create a new database
# False = load the existing database
REBUILD_INDEX = True


# Load local embedding model
print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create or load FAISS database
if os.path.exists(os.path.join(FAISS_PATH, "index.faiss")) and not REBUILD_INDEX:

    print("\nLoading existing FAISS database...")

    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:

    print("\nLoading South Africa PDF...")

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError("PDF file not found!")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Total PDF pages loaded:", len(documents))

    # Clean unusual characters from PDF text
    for doc in documents:
        text = unicodedata.normalize("NFKC", doc.page_content)

        text = (
            text.replace("ʹ", " ")
                .replace("ʥ", "-")
                .replace("ʰ", "'")
                .replace("ʺ", " ")
                .replace("ʭ", " ")
                .replace("ʮ", " ")
                .replace("\u200b", "")
        )

        doc.page_content = text

    # Split PDF into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    print("Total chunks created:", len(chunks))

    # Check whether Wi-Fi information exists
    print("\nChecking Wi-Fi content in PDF...")

    for chunk in chunks:
        clean_text = re.sub(
            r"[^a-z0-9]",
            "",
            chunk.page_content.lower()
        )

        if "wifi" in clean_text:
            print("\nPDF Page:", chunk.metadata.get("page", 0) + 1)
            print(chunk.page_content[:500])

    # Create FAISS vector database
    print("\nCreating FAISS database...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vectorstore.save_local(FAISS_PATH)

    print("\nFAISS database created successfully!")


# Set up retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)


# Connect OpenAI model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# Ask questions using retrieved PDF content
def ask_question(question):

    print("\nSearching FAISS database...")

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        print("No relevant information found.")
        return

    # Display retrieved PDF content
    print("\n===== RETRIEVED DOCUMENTS =====")

    for doc in retrieved_docs:

        page = doc.metadata.get("page", 0) + 1

        print("\nPDF Page:", page)
        print("Content:", doc.page_content[:500])
        print("--------------------------------")

    # Prepare context for OpenAI
    context = ""

    for doc in retrieved_docs:

        page = doc.metadata.get("page", 0) + 1

        context += f"\n[PDF Page {page}]\n"
        context += doc.page_content + "\n"

    # Prepare prompt
    prompt = f"""
    You are a question-answering assistant.

    Answer the question using ONLY the information
    available in the provided PDF context.

    If the answer is not available in the context,
    say that the information is not available in the PDF.

    Do not invent information.

    Mention the relevant PDF page numbers.

    PDF Context:
    {context}

    User Question:
    {question}

    Answer:
    """

    # Generate answer using OpenAI
    print("\nGenerating answer using OpenAI...")

    response = llm.invoke(prompt)

    print("\n========== ANSWER ==========")
    print(response.content)

    # Display retrieved page numbers
    pages = sorted(set(
        doc.metadata.get("page", 0) + 1
        for doc in retrieved_docs
    ))

    print("\nRetrieved PDF pages:", pages)


# Main application
print("\n====================================")
print(" SOUTH AFRICA WIFI - RAG ASSISTANT")
print("====================================")

print("\nAsk questions about the South Africa PDF.")
print("Type 'exit' to stop.")


while True:

    question = input("\nEnter your question: ")

    if question.lower().strip() == "exit":
        print("\nThank you for using the RAG application!")
        break

    if not question.strip():
        print("Please enter a valid question.")
        continue

    ask_question(question)
