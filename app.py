# import os
# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain_community.vectorstores import Pinecone as PineconeStore
# from langchain_groq import ChatGroq
# from pinecone import Pinecone, Index, ServerlessSpec

# def delete_pinecone_index(api_key, index_name):
#     """
#     Deletes all data in the specified Pinecone index if it exists.

#     Parameters:
#     - api_key (str): The API key for Pinecone.
#     - index_name (str): The name of the Pinecone index to delete from.

#     Returns:
#     - None
#     """
#     pc = Pinecone(api_key=api_key)

#     try:
#         # Check if the index exists before attempting to delete
#         if index_name in pc.list_indexes().names():
#             index = pc.Index(index_name)
#             index.delete(delete_all=True)
#             print(f"Successfully deleted all data from index '{index_name}'.")
#     except Exception as e:
#         if "Namespace not found" in str(e):
#             print(f"Index '{index_name}' does not exist. Skipping deletion.")
#         else:
#             print(f"Error deleting index '{index_name}': {str(e)}")

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vectorstore(text_chunks):
#     # Initialize Pinecone client
#     api_key = os.environ.get("PINECONE_API_KEY")
#     pc = Pinecone(api_key=api_key)

#     index_name = "llm"

#     # Create the index if it does not exist
#     if index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=index_name,
#             dimension=768,
#             metric="euclidean",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )

#     # Get the index details and its host
#     index_info = pc.describe_index(index_name)
#     host = index_info.host

#     # Initialize the index with the required api_key and host
#     index = Index(index_name=index_name, api_key=api_key, host=host)

#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")


#     # Create vector store with Pinecone
#     vectorstore = PineconeStore.from_texts(texts=text_chunks, embedding=embeddings, index_name=index_name)
    
#     return vectorstore

# def get_conversation_chain(vectorstore):
#     llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)
#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
    
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
    
#     return conversation_chain

# def handle_userinput(user_question):
#     if st.session_state.conversation is not None:
#         response = st.session_state.conversation({'question': user_question})
#         st.session_state.chat_history = response['chat_history']

#         # Display messages without custom HTML/CSS
#         for i, message in enumerate(st.session_state.chat_history):
#             if i % 2 == 0:
#                 st.write(f"*User:* {message.content}")
#             else:
#                 st.write(f"*Bot:* {message.content}")
#     else:
#         st.warning("Please process the documents first.")

# def main():
#     load_dotenv()
    
#     # Set page config at the beginning of the main function
#     st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    
#     api_key = os.environ.get("PINECONE_API_KEY")
    
#     # Execute delete_pinecone_index only on the initial app load
#     if "index_deleted" not in st.session_state:
#         delete_pinecone_index(api_key=api_key, index_name='llm')
#         st.session_state.index_deleted = True

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs :books:")
    
#     user_question = st.text_input("Ask a question about your documents:")
    
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
        
#         pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # Get PDF text
#                 raw_text = get_pdf_text(pdf_docs)

#                 # Get the text chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # Create vector store
#                 vectorstore = get_vectorstore(text_chunks)

#                 # Create conversation chain
#                 st.session_state.conversation = get_conversation_chain(vectorstore)

# if __name__ == '__main__':
#     main()





























































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
import torch  # Ensure PyTorch is imported

# Set up logging
logging.basicConfig(level=logging.INFO)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        logging.info(f"Processing PDF: {pdf.name}")
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""  # Handle None return from extract_text()
                text += page_text
                logging.debug(f"Extracted text from page: {page_text[:50]}...")  # Log first 50 chars of extracted text
        except Exception as e:
            logging.error(f"Error reading PDF {pdf.name}: {e}")
            st.error(f"Error reading PDF {pdf.name}. Please check the file.")
    logging.info("PDF processing complete.")
    return text

def get_text_chunks(text):
    logging.info("Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Created {len(chunks)} text chunks.")
    return chunks

def get_vectorstore(text_chunks):
    logging.info("Starting to create vector store...")
    
    try:
        # Retrieve Pinecone API key from environment variables
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            logging.error("Pinecone API key is not set.")
            st.error("Pinecone API key is not set. Please check your environment variables.")
            return None
        
        logging.info("Pinecone API key retrieved successfully.")

        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        logging.info("Pinecone client initialized.")

        index_name = "llm"
        logging.info(f"Checking if index '{index_name}' exists...")

        # Check if the index already exists
        existing_indexes = pc.list_indexes().names()
        logging.debug(f"Existing indexes: {existing_indexes}")

        if index_name not in existing_indexes:
            logging.info(f"Index '{index_name}' does not exist. Creating new index...")
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="euclidean",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logging.info(f"Index '{index_name}' created successfully.")
        else:
            logging.info(f"Using existing index: '{index_name}'.")

        # Describe the index to get its details
        index_info = pc.describe_index(index_name)
        host = index_info.host
        logging.info(f"Index '{index_name}' described successfully. Host: {host}")

        # Initialize the Index object
        index = Index(index_name=index_name, api_key=api_key, host=host)
        logging.info(f"Index object created for '{index_name}'.")

        # Initialize embeddings
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        logging.info("HuggingFaceInstructEmbeddings initialized with model 'hkunlp/instructor-xl'.")

        # Create vector store from text chunks
        logging.info(f"Creating vector store from {len(text_chunks)} text chunks...")
        
        vectorstore = PineconeStore.from_texts(texts=text_chunks, embedding=embeddings, index_name=index_name)
        
        logging.info("Vector store created successfully.")
        
        return vectorstore

    except Exception as e:
        logging.error(f"Error creating vector store: {e}", exc_info=True)
        st.error("An error occurred while creating the vector store.")

    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error("An error occurred while creating the vector store.")
        return None

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
    logging.info(f"User question received: {user_question}")
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        
        if 'chat_history' in response:
            st.session_state.chat_history = response['chat_history']
            logging.info("Chat history updated.")

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(f"*User:* {message.content}")
                else:
                    st.write(f"*Bot:* {message.content}")
        else:
            logging.warning("No chat history returned in response.")
    else:
        st.warning("Please process the documents first.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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



# HUGGINGFACEHUB_API_TOKEN=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
# GROQ_API_KEY=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
# PINECONE_API_KEY=7f5896cf-866e-4aeb-8c15-21c3ad004083
