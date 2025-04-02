
# Update your imports at the top of the file:
import streamlit as st
import requests
import time
import os
import tempfile
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
# Import the updated Pinecone SDK
from pinecone import Pinecone

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')  # e.g., 'us-west1-gcp'
pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')    # The name of your index

# Configure the page
st.set_page_config(page_title="Document Query System", layout="wide")

# Apply custom CSS for better UI
st.markdown(
    """
    <style>
        body {
            background-color: #f9f9f9;
            color: #333;
        }
        .chat-bubble {
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 20px;
            max-width: 80%;
        }
        .user-bubble {
            background-color: #2E8B57;
            color: white;
            text-align: right;
            width:fit-content;
            margin-left: auto;
        }
        .ai-bubble {
            background-color: #e0e0e0;
            color: #333;
        }
        .chat-box {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        h2 {
            text-align: center;
            color: #0078D4;
        }
        .submit-button {
            background-color: #0078D4;
            color: white;
            border-radius: 20px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .submit-button:hover {
            background-color: #0056A6;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0078D4;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "source_type" not in st.session_state:
    st.session_state.source_type = None
if "source_urls" not in st.session_state:
    st.session_state.source_urls = {}
if "current_namespace" not in st.session_state:
    st.session_state.current_namespace = None
if "namespace_registry" not in st.session_state:
    st.session_state.namespace_registry = {}  # To track namespaces and their sources
if "pinecone_client" not in st.session_state:
    st.session_state.pinecone_client = None

# Initialize LLM
def initialize_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Initialize Pinecone with updated SDK
def init_pinecone():
    try:
        # Initialize Pinecone client
        if st.session_state.pinecone_client is None:
            st.session_state.pinecone_client = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists
        index_list = st.session_state.pinecone_client.list_indexes()
        index_names = [index.name for index in index_list]
        
        if pinecone_index_name not in index_names:
            st.error(f"Index '{pinecone_index_name}' does not exist. Please create it in the Pinecone dashboard.")
            return False
        return True
    except Exception as e:
        st.error(f"Faied to initialize Pinecone: {str(e)}")
        return False

# Create a unique namespace for a document
def create_namespace(source_name):
    timestamp = int(time.time())
    namespace = f"{source_name.replace(' ', '_').replace('.', '_')}_{timestamp}"
    namespace = namespace[:36].lower()  # Limit length and ensure lowercase
    return namespace

# Delete a namespace when it's no longer needed
def delete_namespace(namespace):
    try:
        if init_pinecone():
            # Get the index
            index = st.session_state.pinecone_client.Index(pinecone_index_name)
            
            # Delete vectors by namespace using the new API
            index.delete(filter={"namespace": namespace})
            return True
    except Exception as e:
        st.error(f"Error deleting namespace: {str(e)}")
    return False

# Web scraping functions
def normalize_url(url):
    """Removes fragment (#) and normalizes the URL."""
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

def get_soup(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser"), response.url
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching {url}: {e}")
        return None, None

def extract_links(soup, base_url):
    """Extract all links from the page, not just nav and footer."""
    links = set()
    for tag in soup.find_all("a"):
        href = tag.get("href")
        if href and not href.startswith(('javascript:', 'mailto:', 'tel:')):
            full_url = normalize_url(urljoin(base_url, href))
            # Only include links from the same domain
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                links.add(full_url)
    return links

def scrape_domain(base_url, max_pages=30, progress_bar=None):
    """Scrape domain with improved link extraction and breadth-first approach."""
    base_domain = urlparse(base_url).netloc
    visited_urls = set()
    to_visit = [base_url]
    scraped_content = []
    url_to_content_map = {}  # Track which URL each chunk came from
    
    page_count = 0
    
    while to_visit and page_count < max_pages:
        current_url = to_visit.pop(0)
        normalized_url = normalize_url(current_url)
        
        if normalized_url in visited_urls:
            continue
            
        visited_urls.add(normalized_url)
        
        soup, final_url = get_soup(current_url)
        if not soup:
            continue
            
        # Get page text content
        page_text = soup.get_text(separator=" ", strip=True)
        
        # Store URL for each content chunk
        url_to_content_map[normalized_url] = page_text
        scraped_content.append(page_text)
        
        # Update progress bar
        page_count += 1
        if progress_bar:
            progress_bar.progress(min(page_count / max_pages, 1.0))
            progress_bar.text(f"Scraped {page_count} pages...")
        
        # Extract links for next pages to visit (breadth-first approach)
        new_links = extract_links(soup, base_url)
        for link in new_links:
            if normalize_url(link) not in visited_urls and normalize_url(link) not in to_visit:
                # Only add links from the same domain
                if urlparse(link).netloc == base_domain:
                    to_visit.append(link)
    
    return scraped_content, url_to_content_map

# Vector DB functions
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name='BAAI/bge-small-en-v1.5', 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

def split_text(text_content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text("\n".join(text_content) if isinstance(text_content, list) else text_content)

def create_documents_with_metadata(text_chunks, url_map=None):
    from langchain_core.documents import Document
    documents = []
    
    if url_map:
        # For web scraped content with URL mapping
        for url, content in url_map.items():
            chunks = split_text(content)
            for chunk in chunks:
                doc = Document(page_content=chunk, metadata={"source": url})
                documents.append(doc)
    else:
        # For PDF content or other sources without URL mapping
        for i, chunk in enumerate(text_chunks):
            doc = Document(page_content=chunk, metadata={"chunk_id": i})
            documents.append(doc)
            
    return documents

def create_vector_db(documents, embeddings, source_name):
    # First make sure Pinecone is initialized
    if not init_pinecone():
        return None
        
    # Create namespace based on source name
    namespace = create_namespace(source_name)
    
    # Save namespace in registry
    st.session_state.namespace_registry[namespace] = {
        "source_name": source_name,
        "created_at": time.time(),
        "type": st.session_state.source_type
    }
    
    # Set as current namespace
    st.session_state.current_namespace = namespace
    
    try:
        # Create vector store with LangchainPinecone (this is from langchain_community.vectorstores)
        vector_store = LangchainPinecone.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=pinecone_index_name,
            namespace=namespace
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector database: {str(e)}")
        return None
# Query functions
def create_prompt_template():
    return ChatPromptTemplate.from_template(
        """
        Answer the question based only on the provided context.
        If you don't know the answer based on the context, say so clearly.
        
        <context>
        {context}
        </context>
        
        Question: {input}
        
        When providing your answer, if the source is available, include a reference 
        to where the information was found.
        """
    )

def process_query(query, vector_store):
    llm = initialize_llm()
    prompt = create_prompt_template()
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input": query})
    
    # Extract source information from retrieved documents
    sources = []
    if "context" in response and hasattr(response["context"], "__iter__"):
        for doc in response["context"]:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                sources.append(doc.metadata["source"])
    
    return response["answer"], list(set(sources))

# Document management sidebar
def document_management_sidebar():
    st.sidebar.header("Document Management")
    
    if st.session_state.namespace_registry:
        st.sidebar.write("Currently loaded documents:")
        
        for namespace, info in st.session_state.namespace_registry.items():
            col1, col2 = st.sidebar.columns([3, 1])
            
            with col1:
                doc_name = info["source_name"]
                doc_type = info["type"].upper()
                is_current = namespace == st.session_state.current_namespace
                
                if is_current:
                    st.info(f"📄 {doc_name} ({doc_type}) - ACTIVE")
                else:
                    st.write(f"📄 {doc_name} ({doc_type})")
            
            with col2:
                # Delete button for this namespace
                if st.button("Delete", key=f"delete_{namespace}"):
                    if delete_namespace(namespace):
                        # Remove from registry
                        del st.session_state.namespace_registry[namespace]
                        
                        # If this was the current namespace, reset
                        if namespace == st.session_state.current_namespace:
                            st.session_state.vector_store = None
                            st.session_state.current_namespace = None
                            st.session_state.chat_history = []
                        
                        st.success(f"Deleted {doc_name}")
                        st.experimental_rerun()
                
                # Switch to this document if it's not current
                if not is_current:
                    if st.button("Switch", key=f"switch_{namespace}"):
                        # Connect to existing namespace
                        embeddings = create_embeddings()
                        st.session_state.vector_store = Pinecone(
                            index_name=pinecone_index_name,
                            embedding=embeddings,
                            namespace=namespace
                        )
                        st.session_state.current_namespace = namespace
                        
                        # Reset chat history when switching documents
                        st.session_state.chat_history = []
                        
                        st.success(f"Switched to {doc_name}")
                        st.experimental_rerun()
    else:
        st.sidebar.write("No documents loaded yet.")
        
    # Add a button to clear all namespaces
    if st.session_state.namespace_registry and st.sidebar.button("Clear All Documents"):
        for namespace in list(st.session_state.namespace_registry.keys()):
            delete_namespace(namespace)
        
        # Reset all state
        st.session_state.namespace_registry = {}
        st.session_state.vector_store = None
        st.session_state.current_namespace = None
        st.session_state.chat_history = []
        
        st.sidebar.success("All documents cleared")
        st.experimental_rerun()

# Main app interface
st.title("Document Query System")
st.write("Upload a PDF or enter a domain to scrape, then ask questions about the content.")

# Check if Pinecone environment variables are set
if not pinecone_api_key or not pinecone_environment or not pinecone_index_name:
    st.error("Pinecone API key, environment, or index name not set. Please add these to your .env file.")
    st.code("""
    # Add to your .env file:
    PINECONE_API_KEY=your_api_key
    PINECONE_ENVIRONMENT=your_environment  # e.g., us-west1-gcp
    PINECONE_INDEX_NAME=your_index_name
    """)
    st.stop()

# Add document management sidebar
document_management_sidebar()

# Input method tabs
tab1, tab2 = st.tabs(["Domain Scraping", "PDF Upload"])

with tab1:
    st.header("Domain Scraping")
    domain_input = st.text_input("Enter a domain to scrape (e.g., https://example.com)")
    max_pages = st.slider("Maximum pages to scrape", 5, 100, 30)
    
    if st.button("Start Scraping", key="scrape_button"):
        if domain_input:
            st.session_state.source_type = "web"
            
            # Extract domain name for namespace
            domain_name = urlparse(domain_input).netloc
            
            progress_bar = st.progress(0)
            progress_text = st.empty()
            progress_text.text("Starting scraping...")
            
            scraped_content, url_to_content_map = scrape_domain(domain_input, max_pages, progress_bar)
            st.session_state.source_urls = url_to_content_map
            
            progress_text.text("Creating embeddings...")
            embeddings = create_embeddings()
            
            progress_text.text("Creating documents...")
            documents = create_documents_with_metadata(scraped_content, url_to_content_map)
            
            progress_text.text("Creating vector database in Pinecone...")
            # Pass domain name as source name
            st.session_state.vector_store = create_vector_db(documents, embeddings, domain_name)
            
            if st.session_state.vector_store:
                progress_bar.progress(1.0)
                progress_text.text(f"Done! Scraped {len(url_to_content_map)} pages from {domain_input}")
                st.success(f"Successfully scraped and processed {domain_input}")
                
                # Reset chat history for new document
                st.session_state.chat_history = []
                
                st.experimental_rerun()
            else:
                progress_bar.progress(1.0)
                st.error("Failed to create vector database. Check Pinecone configuration.")
        else:
            st.error("Please enter a valid domain")

with tab2:
    st.header("PDF Upload")
    pdf_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    
    if st.button("Process PDF", key="pdf_button") and pdf_file:
        st.session_state.source_type = "pdf"
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Save uploaded PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.read())
            pdf_path = temp_file.name
        
        progress_bar.progress(0.2)
        progress_text.text("Loading PDF...")
        
        # Load and process PDF
        loader = PyPDFLoader(pdf_path)
        pdf_documents = loader.load()
        
        progress_bar.progress(0.4)
        progress_text.text("Creating embeddings...")
        
        embeddings = create_embeddings()
        
        progress_bar.progress(0.6)
        progress_text.text("Splitting content...")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        pdf_chunks = text_splitter.split_documents(pdf_documents)
        
        progress_bar.progress(0.8)
        progress_text.text("Creating vector database in Pinecone...")
        
        # Store PDF metadata including page numbers
        for chunk in pdf_chunks:
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = f"Page {chunk.metadata.get('page', 'unknown')} of {pdf_file.name}"
        
        # Use PDF filename as source name
        st.session_state.vector_store = create_vector_db(pdf_chunks, embeddings, pdf_file.name)
        
        if st.session_state.vector_store:
            st.session_state.source_urls = {f"Page {doc.metadata.get('page', i)}": pdf_file.name for i, doc in enumerate(pdf_chunks)}
            
            progress_bar.progress(1.0)
            progress_text.text("Done!")
            
            # Clean up the temp file
            os.unlink(pdf_path)
            
            # Reset chat history for new document
            st.session_state.chat_history = []
            
            st.success(f"Successfully processed PDF: {pdf_file.name}")
            st.experimental_rerun()
        else:
            progress_bar.progress(1.0)
            st.error("Failed to create vector database. Check Pinecone configuration.")

# Query interface (only show if vector store exists)
if st.session_state.vector_store:
    active_doc = "unknown"
    for namespace, info in st.session_state.namespace_registry.items():
        if namespace == st.session_state.current_namespace:
            active_doc = info["source_name"]
    
    st.header(f"Ask Questions about: {active_doc}")
    
    # Display what source was processed
    if st.session_state.source_type == "web":
        st.info(f"You can now ask questions about the scraped website.")
    elif st.session_state.source_type == "pdf":
        st.info(f"You can now ask questions about the uploaded PDF.")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        for user_input, ai_response, sources in st.session_state.chat_history:
            st.markdown(f"<div class='chat-bubble user-bubble'>{user_input}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble ai-bubble'>{ai_response}</div>", unsafe_allow_html=True)
            if sources:
                st.markdown("<div style='font-size: 0.8em; color: #666; margin-left: 10px;'>Sources: " + 
                           ", ".join([f"<a href='{s}' target='_blank'>{s[:50]}...</a>" if s.startswith('http') else s for s in sources]) + 
                           "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Query input
    user_query = st.text_input("Enter your question")
    
    if st.button("Submit Question", key="query_button"):
        if user_query:
            with st.spinner("Processing your question..."):
                answer, sources = process_query(user_query, st.session_state.vector_store)
                
                # Update chat history
                st.session_state.chat_history.append((user_query, answer, sources))
                
                # Force refresh to display new messages
                st.experimental_rerun()
        else:
            st.error("Please enter a question")
else:
    st.info("Please process a document first to start asking questions")