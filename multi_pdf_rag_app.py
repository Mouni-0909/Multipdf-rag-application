
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


# Step 1: Load environment variables from .env
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing in .env file")


# Step 2: Set folder paths
BASE_DIR = Path(__file__).parent

PDF_FOLDER = BASE_DIR / "pdfs"

# Separate database for the multi-PDF application
FAISS_PATH = BASE_DIR / "multi_pdf_faiss_index"

# Set True for the first run or whenever PDFs change
REBUILD_INDEX = False


# Step 3: List all three PDF files
PDF_FILES = [
    PDF_FOLDER / "internetsetup.pdf",
    PDF_FOLDER / "javascript_tutorial.pdf",
    PDF_FOLDER / "PYTHON PROGRAMMING NOTES.pdf"
]


# Step 4: Check whether all PDF files exist
for pdf_file in PDF_FILES:
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")


# Step 5: Load PDF documents
documents = []

print("\nLoading PDF files...")

for pdf_file in PDF_FILES:

    loader = PyPDFLoader(str(pdf_file))
    pdf_documents = loader.load()

    # Store the PDF filename in document metadata
    for doc in pdf_documents:
        doc.metadata["source"] = pdf_file.name

    documents.extend(pdf_documents)

    print(f"Loaded: {pdf_file.name} - {len(pdf_documents)} pages")


print(f"\nTotal PDF files: {len(PDF_FILES)}")
print(f"Total PDF pages: {len(documents)}")


# Step 6: Clean extracted PDF text
def clean_text(text):

    unwanted_characters = ["ʹ", "ʥ", "ʰ", "ʺ", "ʭ", "ʮ", "\u200b"]

    for character in unwanted_characters:
        text = text.replace(character, "")

    return text


for doc in documents:
    doc.page_content = clean_text(doc.page_content)


# Step 7: Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print(f"\nTotal text chunks created: {len(chunks)}")


# Step 8: Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 9: Create or load FAISS vector database
if REBUILD_INDEX or not FAISS_PATH.exists():

    print("\nCreating multi-PDF FAISS database...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(str(FAISS_PATH))

    print("FAISS database created successfully!")

else:

    print("\nLoading existing FAISS database...")

    vectorstore = FAISS.load_local(
        str(FAISS_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("FAISS database loaded successfully!")


# Step 10: Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)


# Step 11: Initialize OpenAI model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# Step 12: Ask questions using all PDFs
def ask_question(question):

    print("\nSearching all PDF documents...")

    retrieved_docs = retriever.invoke(question)

    context = ""
    source_pages = []

    # Collect retrieved text and source details
    for doc in retrieved_docs:

        filename = doc.metadata.get("source", "Unknown PDF")
        page_number = doc.metadata.get("page", 0) + 1

        context += (
            f"\nPDF: {filename}\n"
            f"Page: {page_number}\n"
            f"Content: {doc.page_content}\n"
        )

        source_pages.append((filename, page_number))

    # Create prompt for OpenAI
    prompt = f"""
    You are a helpful assistant.

    Answer the question using only the information
    provided in the PDF context below.

    If the answer is not available in the PDFs,
    clearly say that the information is not available
    in the uploaded documents.

    Mention the PDF filename and page number
    wherever possible.

    PDF Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Generate answer
    response = llm.invoke(prompt)

    print("\n========== ANSWER ==========")
    print(response.content)

    # Display retrieved PDF pages
    print("\n========== SOURCE PAGES ==========")

    for filename, page_number in sorted(set(source_pages)):
        print(f"{filename} - Page {page_number}")


# Step 13: Keep asking questions until the user exits
print("\n====================================")
print("       MULTI-PDF RAG APPLICATION")
print("====================================")

print("\nAvailable PDF documents:")

for pdf_file in PDF_FILES:
    print("-", pdf_file.name)

while True:

    question = input("\nEnter your question (type 'exit' to stop): ")

    if question.lower() == "exit":
        print("\nThank you for using the Multi-PDF RAG App!")
        break

    if question.strip():
        ask_question(question)
