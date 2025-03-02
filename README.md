# Chat with Multiple PDFs

This Streamlit application allows users to upload multiple PDF documents and engage in a conversational Q&A with the content of those documents. It leverages LangChain, Ollama, and Sentence Transformers to provide a seamless and interactive experience.

## Features

-   **Upload Multiple PDFs:** Users can upload multiple PDF documents for processing.
-   **Conversational Q&A:** Engage in a natural language conversation with the content of the uploaded PDFs.
-   **LLM-Powered Answers:** Utilizes Ollama and LangChain for powerful and context-aware responses.
-   **Vector Store for Efficient Retrieval:** Employs FAISS and Sentence Transformers for efficient semantic search within the documents.
-   **Chat History:** Maintains a continuous chat history for a smooth conversational flow.
-   **Clean and User-Friendly Interface:** Built with Streamlit for an intuitive and responsive user experience.

## Code Explanation

This Streamlit application facilitates a conversational Q&A system using uploaded PDF documents. Here's a detailed breakdown of how the code works:

1.  **PDF Text Extraction (`get_pdf_text`):**
    * The `get_pdf_text` function takes a list of uploaded PDF documents (`pdf_docs`) as input.
    * It iterates through each PDF, reads its pages using `PyPDF2.PdfReader`, and extracts the text content from each page.
    * The extracted text from all PDFs is concatenated into a single string and returned.

2.  **Text Chunking (`get_text_chunks`):**
    * The `get_text_chunks` function receives the combined text from the PDFs.
    * It uses LangChain's `CharacterTextSplitter` to divide the text into smaller, manageable chunks.
    * The text is split based on newline characters (`\n`), with a specified `chunk_size` and `chunk_overlap` to ensure context is maintained between chunks.
    * The resulting list of text chunks is returned.

3.  **Vector Store Creation (`get_vectorstore`):**
    * The `get_vectorstore` function takes the list of text chunks as input.
    * It initializes a `SentenceTransformerEmbeddings` model to generate embeddings for the text chunks.
    * FAISS (Facebook AI Similarity Search) is used to create a vector store from the text chunks and their embeddings, enabling efficient semantic search.
    * If there is an error during vectorstore creation, an error message is displayed to the user.
    * The created vector store is returned.

4.  **Language Model Initialization (`get_llm`):**
    * The `get_llm` function initializes the Ollama language model.
    * It uses `ChatOllama` to instantiate the model, in this case, `llama2:7b`.
    * If there is an error during LLM initialization, an error message is displayed to the user.
    * The initialized LLM is returned.

5.  **Conversation Chain Setup (`get_conversation_chain`):**
    * The `get_conversation_chain` function sets up the conversational Q&A system.
    * It initializes a `ConversationBufferMemory` to store the chat history, allowing the LLM to maintain context across multiple turns.
    * LangChain's `ConversationalRetrievalChain` is used to create the conversation chain, combining the LLM, vector store, and memory.
    * The `return_source_documents` parameter is set to false to prevent source documents from being returned in the response.
    * The created conversation chain is returned.

6.  **Answer Display (`display_answer`):**
    * The `display_answer` function takes the LLM's response as input.
    * It extracts the answer from the response and appends it to the `st.session_state.messages` list, which stores the chat history.

7.  **Question Processing (`process_question`):**
    * The `process_question` function takes the users prompt as an argument.
    * If a conversation chain exists, the function sends the users question to the conversation chain.
    * The response from the conversation chain is then passed to the `display_answer` function.

8.  **Main Application (`main`):**
    * The `main` function sets up the Streamlit application.
    * It initializes session state variables to store the conversation chain and chat history.
    * It creates a sidebar for uploading PDF documents and processing them.
    * When the "Process" button is clicked, the PDF text is extracted, chunked, and a vector store is created. The conversation chain is then initialized and stored in session state.
    * The chat history is displayed using `st.chat_message` to create a chat-like interface.
    * The `st.chat_input` function takes the users question.
    * The `process_question` function is called as a callback when the user submits a question.
    * The users question is then displayed.


## Prerequisites

Before running the application, ensure you have the following installed:

-   Python 3.7+
-   Streamlit
-   LangChain
-   Ollama
-   Sentence Transformers
-   PyPDF2
-   python-dotenv

You can install the required packages using pip:

```bash
pip install streamlit langchain ollama sentence-transformers PyPDF2 python-dotenv