"""
coding   : utf-8
@Desc    : Main Function
@Time    : 2023/9/5 10:36
@Author  : KaiVen
@Email   : herofolk@outlook.com
@File    : main.py
"""


import os
import shutil
import time

# Modules to Import
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, AzureOpenAI, CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import tempfile
from typing import List
from utils import *


@st.cache_resource
def get_qa(choose_model):
    # Create large language model and embeddings
    llm = get_model(choose_model)
    embeddings = get_embeddings(choose_model)
    # Save the generated embedding into a Chroma database for storage and easy retrieval
    if os.path.exists('db'):
        # delete the path
        shutil.rmtree('db')
    vector = get_vector()
    # Default save on local disk
    db = vector.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
    # Create a RetrievalQA
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={"k": 3}),
                                     return_source_documents=True, verbose=False)
    return embeddings, db, llm, qa


@st.cache_data
def save_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    return tmp_file_path


@st.cache_data
def parse_loader(tmp_path: str, file_name: str) -> List[Document]:
    ext = "." + file_name.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        return loader_class(tmp_path, **loader_args)
    else:
        raise "The file type does not support parsing!"


@st.cache_data
def parse_file(uploaded_file, tmp_file_path):
    name_of_file = uploaded_file.name
    # Loader from file
    loader = parse_loader(tmp_file_path, name_of_file)
    # Load Documents and split into chunks
    documents = loader.load_and_split()
    # Chunks the text in 1024 characters and adds an overlap of 64 characters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    return name_of_file, texts


# Web configuration.
st.title("🤖 Personalized Bot with File 🧠 ")
st.markdown(
    """ 
        ####  🗨️ Chat with your files 📜  
        > *powered by [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [OpenAI]('https://platform.openai.com/docs/models/gpt-3-5') +
        [Streamlit]('https://streamlit.io/')*
        ----
        """
)
st.sidebar.markdown(
    """
    ### Steps:
    1. Enter Your Secret Key for Embeddings
    2. Upload Your File for example pdf
    3. Perform Q&A
**Note : File content and API key not stored in any form.**
    """
)

# First configure your model

option = st.selectbox('Pick One Model', ['gpt4all', 'azure-openai'])
time.sleep(5)
if option:
    # choose the model and embedding model name
    st.write('You selected:', option)
    # load the user file
    uploaded_file = st.file_uploader('Upload your Files here.', type=['csv', 'pdf', 'doc'])
    if uploaded_file:
        with st.spinner('Uploading'):
            tmp_file_path = save_file(uploaded_file)
        st.success('Load the file done.')
        with st.spinner('Processing of parse the file.'):
            name_of_file, texts = parse_file(uploaded_file, tmp_file_path)
            st.success('Parse the file done.')
        # get Q&A and llm
        with st.spinner('Processing of load the llm.'):
            embeddings, db, llm, qa = get_qa(option)
            st.success('Load the llm done.')
        query = st.text_input(
            "**What's on your mind?**",
            placeholder='Ask me anything from {}'.format(name_of_file),
        )
        if query:
            with st.spinner('Generating Answer to your Query : `{}` '.format(query)):
                res = qa(query)
                st.info(res['result'], icon="🤖")
