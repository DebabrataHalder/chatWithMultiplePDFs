import os
import logging
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_groq import ChatGroq
from pinecone import Pinecone, Index, ServerlessSpec

# Set up logging
logging.basicConfig(level=logging.INFO)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        logging.info(f"Processing PDF: {pdf.name}")
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Handle None return from extract_text()
        except Exception as e:
            logging.error(f"Error reading PDF {pdf.name}: {e}")
            st.error(f"Error reading PDF {pdf.name}. Please check the file.")
    logging.info("PDF processing complete.")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Created {len(chunks)} text chunks.")
    return chunks

def clear_index(index_name, api_key):
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # Check if the index exists
        if index_name in pc.list_indexes().names():
            # Get the host information
            index_info = pc.describe_index(index_name)
            host = index_info.host
            
            # Initialize the index with both api_key and host
            index = Index(index_name=index_name, api_key=api_key, host=host)
            
            # Clear all content in the index
            index.delete(delete_all=True)
            logging.info(f"Cleared all content from index '{index_name}'.")
    except Exception as e:
        logging.error(f"Error clearing index '{index_name}': {e}")

def get_vectorstore(text_chunks):
    try:
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = "pdfchat"

        # Initialize Pinecone and create the index if it doesn't exist
        pc = Pinecone(api_key=api_key)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="euclidean",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        index_info = pc.describe_index(index_name)
        host = index_info.host

        index = Index(index_name=index_name, api_key=api_key, host=host)

        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

        vectorstore = PineconeStore.from_texts(texts=text_chunks, embedding=embeddings, index_name=index_name)
        
        logging.info("Vector store created successfully.")
        return vectorstore

    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error("An error occurred while creating the vector store.")

def get_conversation_chain(vectorstore):
    try:
        llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        
        logging.info("Conversation chain created successfully.")
        return conversation_chain
    
    except Exception as e:
        logging.error(f"Error creating conversation chain: {e}")
        st.error("An error occurred while setting up the conversation chain.")

def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"*User:* {message.content}")
            else:
                st.write(f"*Bot:* {message.content}")
    else:
        st.warning("Please process the documents first.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "index_cleared" not in st.session_state:
        st.session_state.index_cleared = False

    # Clear index only once when the app is first loaded
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = "pdfchat"
    if not st.session_state.index_cleared:
        clear_index(index_name, api_key)
        st.session_state.index_cleared = True

    st.header("Chat with multiple PDFs :books:")
    
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    try:
                        # Get PDF text
                        raw_text = get_pdf_text(pdf_docs)

                        # Get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # Create conversation chain
                        if vectorstore is not None:  # Ensure vectorstore was created successfully
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                    except Exception as e:
                        logging.error(f"Error during processing: {e}")
                        st.error("An error occurred during processing. Please try again.")
            else:
                st.warning("Please upload at least one PDF document.")

if __name__ == '__main__':
    main()
