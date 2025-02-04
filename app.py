import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import hashlib
from datetime import datetime

# --- Load Environment Variables ---
if not load_dotenv():
    st.error("❌ Could not load .env file. Ensure it exists in the same directory as this script.")
    st.stop()


# --- Validate API Keys ---
def get_env_var(key, name):
    value = os.getenv(key)
    if not value:
        st.error(f"❌ Missing {name} API key in .env file! (Expected key: {key})")
        st.stop()
    return value


PINECONE_API_KEY = get_env_var("PINECONE_API_KEY", "Pinecone")
GROQ_API_KEY = get_env_var("GROQ_API_KEY", "Groq")
HUGGINGFACEHUB_API_TOKEN = get_env_var("HUGGINGFACEHUB_API_TOKEN", "Hugging Face")

# --- Initialize Pinecone ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    st.error(f"⚠️ Failed to initialize Pinecone: {e}")
    st.stop()

# --- Initialize LLM ---
try:
    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0,
        api_key=GROQ_API_KEY
    )
except Exception as e:
    st.error(f"⚠️ Failed to initialize Groq LLM: {e}")
    st.stop()

# --- Initialize Embeddings ---
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
except Exception as e:
    st.error(f"⚠️ Failed to initialize embeddings: {e}")
    st.stop()

# --- Constants ---
MAIN_INDEX_NAME = "pdf-analyzer-main-index"  # Single index for all documents
DIMENSION = 768  # Dimension for all-mpnet-base-v2 embeddings


# --- Helper Functions ---
def create_namespace(file_name):
    """Create a unique namespace for each document."""
    file_hash = hashlib.md5(file_name.encode()).hexdigest()[:16]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{file_hash}-{timestamp}"


def split_text(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def get_or_create_index():
    """Get or create the main Pinecone index."""
    if MAIN_INDEX_NAME not in pc.list_indexes().names():
        with st.spinner("🌲 Creating Pinecone index..."):
            try:
                pc.create_index(
                    name=MAIN_INDEX_NAME,
                    dimension=DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            except Exception as e:
                st.error(f"⚠️ Failed to create Pinecone index: {e}")
                st.error(
                    "You've reached the maximum number of serverless indexes (5). Please delete unused indexes in the Pinecone console.")
                st.stop()
    return pc.Index(MAIN_INDEX_NAME)


# --- Streamlit App ---
st.title("📄 Smart PDF Analyzer with AI")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    try:
        # Save uploaded file temporarily
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create unique namespace
        namespace = create_namespace(uploaded_file.name)

        # Process PDF
        with st.spinner("📄 Processing PDF..."):
            loader = PyPDFLoader(temp_file)
            pages = loader.load()
            chunks = split_text(pages)

        # Get or create Pinecone index
        index = get_or_create_index()

        # Store embeddings in Pinecone
        with st.spinner("📊 Indexing document..."):
            PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                index_name=MAIN_INDEX_NAME,
                namespace=namespace,
            )

        # Initialize retriever
        vector_store = PineconeVectorStore(
            index_name=MAIN_INDEX_NAME,
            embedding=embeddings,
            pinecone_client=pc,  # Pass the initialized Pinecone client
            namespace=namespace
        )

        st.session_state.retriever = vector_store.as_retriever(
            search_kwargs={"k": 3, "namespace": namespace}
        )

        st.success(f"✅ Document processed and stored in namespace '{namespace[:15]}...'!")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Query Interface
if "retriever" in st.session_state:
    st.header("Ask Questions")
    query = st.text_input("Enter your question about the document:")

    if query:
        with st.spinner("🤖 Generating answer..."):
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.retriever,
                    return_source_documents=True
                )

                response = qa_chain.invoke({"query": query})

                st.subheader("Answer:")
                st.write(response["result"])

                with st.expander("View source passages"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Passage {i + 1}** (Page {doc.metadata['page'] + 1})")
                        st.text(doc.page_content[:500] + "...")

            except Exception as e:
                st.error(f"⚠️ Query error: {e}")

else:
    st.info("👈 Upload a PDF to get started")