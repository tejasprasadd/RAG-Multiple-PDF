import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import SentenceTransformerEmbeddings

load_dotenv()

# --- Helper Functions ---

# Extract text from uploaded PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split the extracted text into smaller chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vector store from text chunks for efficient retrieval
def get_vectorstore(text_chunks):
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

# Initialize the language model
def get_llm():
    try:
        llm = ChatOllama(model="llama2:7b")
        return llm
    except Exception as e:
        st.error(f"Error initializing Ollama: {e}")
        return None

# Create a conversation chain for interactive Q&A
def get_conversation_chain(vectorstore, llm):
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key="answer"
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain

# --- Main Streamlit App ---

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.markdown("<h1 style='text-align: center;'>Chat with Multiple PDFs</h1>", unsafe_allow_html=True)

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

    # Sidebar for document uploads
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on Process", accept_multiple_files=True)
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Mugging Up the PDFs"):
                    # Clear previous messages when processing new documents
                    st.session_state.messages = []
                    
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    
                    if vectorstore is not None:
                        llm = get_llm()
                        if llm is not None:
                            st.session_state.conversation = get_conversation_chain(vectorstore, llm)
                            st.session_state.processing_complete = True
                            st.success("Processing complete! You can now ask questions about your documents.")
            else:
                st.error("Please upload PDF documents first.")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response if conversation is initialized
        if st.session_state.conversation:
            with st.chat_message("assistant"):
                with st.spinner("Scratching Head Vigorously"):
                    # Get response from the model
                    response = st.session_state.conversation({"question": prompt})
                    answer = response.get('answer', "I couldn't find an answer to that question in the provided documents.")
                    
                    # Format source information in a cleaner way
                    source_docs = response.get('source_documents', [])
                    if source_docs:
                        # Display the answer first
                        st.markdown(answer)
                        
                        # Add a collapsible section for sources if available
                        with st.expander("Sources"):
                            for i, doc in enumerate(source_docs):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                    else:
                        # Just display the answer if no sources
                        st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            # Display a message if documents haven't been processed yet
            with st.chat_message("assistant"):
                if not st.session_state.processing_complete:
                    error_msg = "Please upload and process documents first before asking questions."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    error_msg = "I'm having trouble accessing the documents. Please try processing them again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == '__main__':
    main()