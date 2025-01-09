import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()




st.set_page_config(page_title="PDF Chat", layout="wide")

# Initialize Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Store in streamlit secrets
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",
    temperature=0.7,
    max_tokens=2048
)



# Initialize Neo4j Graph
NEO4J_URI = os.getenv("NEO4J_URI") # Update with your Neo4j URI
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")  # Update with your username
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # Update with your password
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    
    graph.query("""
        CREATE CONSTRAINT document_id IF NOT EXISTS
        FOR (d:Document) REQUIRE d.id IS UNIQUE
    """)
    
    vectorstore = Neo4jVector.from_documents(
        documents=texts,
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="pdf_index",
        node_label="Document",
        text_node_property="text",
        embedding_node_property="embedding"
    )
    
    os.unlink(tmp_path)
    return vectorstore, len(texts)

def init_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Specify the output key explicitly
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=True,
        return_source_documents=True
    )
    
    return conversation_chain

# Custom CSS
st.markdown("""
    <style>
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #2b313e;
        }
        .chat-message.assistant {
            background-color: #475063;
        }
        .chat-message .message {
            color: #ffffff;
            font-size: 1rem;
        }
        .source {
            font-size: 0.85rem;
            color: #a8a8a8;
            margin-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Main interface
st.title("📚 PDF Chat with Graph RAG")

# Sidebar
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    
    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("Processing PDF..."):
            vectorstore, num_chunks = process_pdf(uploaded_file)
            st.session_state.conversation = init_conversation_chain(vectorstore)
            st.session_state.pdf_processed = True
            st.success(f"✅ PDF processed into {num_chunks} chunks!")
    
    if st.session_state.pdf_processed:
        st.header("🔄 Reset Chat")
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.conversation = init_conversation_chain(vectorstore)
            st.success("Conversation cleared!")

# Main chat interface
if not st.session_state.pdf_processed:
    st.info("👈 Please upload a PDF document to start chatting!")
else:
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.container():
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        <div class="message">{message['content']}</div>
                        {f'<div class="source">{message["source"]}</div>' if "source" in message else ""}
                    </div>
                """, unsafe_allow_html=True)

    # Chat input
    user_question = st.chat_input("Ask a question about your document...")
    if user_question:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        with st.spinner("Thinking..."):
            # Get response
            response = st.session_state.conversation({
                "question": user_question
            })
            
            # Process sources
            sources = ""
            if response.get("source_documents"):
                sources = "Sources: " + ", ".join([
                    f"Chunk {i+1}" for i, _ in enumerate(response["source_documents"])
                ])
            
            # Add assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"],
                "source": sources
            })
        
        # Update display
        st.rerun()