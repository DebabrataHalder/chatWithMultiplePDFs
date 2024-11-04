import os
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

def delete_pinecone_index(api_key, index_name):
    """
    Deletes all data in the specified Pinecone index if it exists.

    Parameters:
    - api_key (str): The API key for Pinecone.
    - index_name (str): The name of the Pinecone index to delete from.

    Returns:
    - None
    """
    pc = Pinecone(api_key=api_key)

    try:
        # Check if the index exists before attempting to delete
        if index_name in pc.list_indexes().names():
            index = pc.Index(index_name)
            index.delete(delete_all=True)
            print(f"Successfully deleted all data from index '{index_name}'.")
    except Exception as e:
        if "Namespace not found" in str(e):
            print(f"Index '{index_name}' does not exist. Skipping deletion.")
        else:
            print(f"Error deleting index '{index_name}': {str(e)}")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Initialize Pinecone client
    api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    index_name = "llm"

    # Create the index if it does not exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Get the index details and its host
    index_info = pc.describe_index(index_name)
    host = index_info.host

    # Initialize the index with the required api_key and host
    index = Index(index_name=index_name, api_key=api_key, host=host)

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")


    # Create vector store with Pinecone
    vectorstore = PineconeStore.from_texts(texts=text_chunks, embedding=embeddings, index_name=index_name)
    
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # Display messages without custom HTML/CSS
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"*User:* {message.content}")
            else:
                st.write(f"*Bot:* {message.content}")
    else:
        st.warning("Please process the documents first.")

def main():
    load_dotenv()
    
    # Set page config at the beginning of the main function
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    
    api_key = os.environ.get("PINECONE_API_KEY")
    
    # Execute delete_pinecone_index only on the initial app load
    if "index_deleted" not in st.session_state:
        delete_pinecone_index(api_key=api_key, index_name='llm')
        st.session_state.index_deleted = True

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
