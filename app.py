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






# aiohappyeyeballs==2.4.3
# aiohttp==3.10.10
# aiosignal==1.3.1
# altair==4.0.0
# annotated-types==0.7.0
# anyio==4.6.2.post1
# async-timeout==4.0.3
# attrs==24.2.0
# blinker==1.8.2
# cachetools==5.5.0
# certifi==2024.8.30
# charset-normalizer==3.4.0
# click==8.1.7
# colorama==0.4.6
# dataclasses-json==0.5.14
# distro==1.9.0
# entrypoints==0.4
# exceptiongroup==1.2.2
# faiss-cpu==1.7.4
# filelock==3.16.1
# frozenlist==1.5.0
# fsspec==2024.10.0
# gitdb==4.0.11
# GitPython==3.1.43
# greenlet==3.1.1
# groq==0.11.0
# h11==0.14.0
# httpcore==1.0.6
# httpx==0.27.2
# huggingface-hub==0.14.1
# idna==3.10
# importlib_metadata==8.5.0
# InstructorEmbedding==1.0.1
# Jinja2==3.1.4
# joblib==1.4.2
# jsonpatch==1.33
# jsonpointer==3.0.0
# jsonschema==4.23.0
# jsonschema-specifications==2024.10.1
# langchain==0.3.4
# langchain-community==0.3.3
# langchain-core==0.3.13
# langchain-groq==0.2.0
# langchain-text-splitters==0.3.0
# langsmith==0.1.137
# markdown-it-py==3.0.0
# MarkupSafe==3.0.2
# marshmallow==3.23.0
# mdurl==0.1.2
# mpmath==1.3.0
# multidict==6.1.0
# mypy-extensions==1.0.0
# networkx==3.4.2
# nltk==3.9.1
# numexpr==2.10.1
# numpy==1.26.4
# openai==0.27.6
# openapi-schema-pydantic==1.2.4
# orjson==3.10.10
# packaging==24.1
# pandas==2.2.3
# pillow==11.0.0
# pinecone==5.3.1
# pinecone-plugin-inference==1.1.0
# pinecone-plugin-interface==0.0.7
# propcache==0.2.0
# protobuf==3.20.3
# pyarrow==18.0.0
# pydantic==2.9.2
# pydantic-settings==2.6.0
# pydantic_core==2.23.4
# pydeck==0.9.1
# Pygments==2.18.0
# Pympler==1.1
# PyPDF2==3.0.1
# python-dateutil==2.9.0.post0
# python-dotenv==1.0.0
# pytz==2024.2
# pywin32==308
# PyYAML==6.0.2
# referencing==0.35.1
# regex==2024.9.11
# requests==2.32.3
# requests-toolbelt==1.0.0
# rich==13.9.3
# rpds-py==0.20.0
# safetensors==0.4.5
# scikit-learn==1.5.2
# scipy==1.14.1
# semver==3.0.2
# sentence-transformers==2.2.2
# sentencepiece==0.2.0
# six==1.16.0
# smmap==5.0.1
# sniffio==1.3.1
# SQLAlchemy==2.0.36
# streamlit==1.18.1
# sympy==1.13.1
# tenacity==8.5.0
# threadpoolctl==3.5.0
# tiktoken==0.4.0
# tokenizers==0.13.3
# toml==0.10.2
# toolz==1.0.0
# torch==2.5.0
# torchvision==0.20.0
# tornado==6.4.1
# tqdm==4.66.6
# transformers==4.31.0
# typing-inspect==0.9.0
# typing_extensions==4.12.2
# tzdata==2024.2
# tzlocal==5.2
# urllib3==2.2.3
# validators==0.34.0
# watchdog==5.0.3
# yarl==1.16.0
# zipp==3.20.2






# HUGGINGFACEHUB_API_TOKEN=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
# GROQ_API_KEY=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
# PINECONE_API_KEY=7f5896cf-866e-4aeb-8c15-21c3ad004083