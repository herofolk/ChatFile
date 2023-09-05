"""
coding   : utf-8
@Desc    : The general functions or classes library by user.
@Time    : 2023/9/5 10:38
@Author  : KaiVen
@Email   : herofolk@outlook.com
@File    : utils.py
"""

# public library
from typing import Any, Optional, List, Dict
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.llms import GPT4All, AzureOpenAI, CTransformers
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel
from langchain.vectorstores import Chroma, FAISS
from langchain.callbacks.manager import CallbackManagerForLLMRun
import os
import torch
from langchain.document_loaders import *

# the map of loader list
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".xls": (UnstructuredExcelLoader, {}),
    # Add more mappings for other file extensions and loaders as needed
}

# use to calculate device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# large language model configuration
LLAMA_CONFIG = {'max_new_tokens': 1024, 'temperature': 0.01}
# local save directory to database function use
PERSIST_DIRECTORY = 'vector_store_path'
MODEL_NAME = 'GPT4All'
# embedding the file text
TRANSFORMERS_PATH = r'../models/sentence_transformers'
EMBEDDINGS_MODEL = 'all-MiniLM-L6-v2'

# list of supported large language models
SUPPORT_MODEL_LIST = ['gpt4all', 'azure-openai', 'llama', 'llama2', 'chat-glm', 'chat-glm2']

# the path of local models
GPT4ALL = r'../models/ggml-gpt4all-j-v1.3-groovy.bin'
LLAMA = r'../models/llama-7b.ggmlv3.q4_1.bin'
LLAMA2 = r'../models/llama-2-7b.ggmlv3.q4_1.bin'
ChatGLM = r'..\models\chatglm-6b-int4'
ChatGLM2 = r'..\models\chatglm2-6b-int4'

# default local file path
DEFAULT_FILE_PATH = '../files/ChatWithFile.pdf'

# default azure-openai configuration
os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_VERSION'] = '2023-03-15-preview'
os.environ['OPENAI_API_BASE'] = 'https://esq-chatgpt-pg.openai.azure.com/'
os.environ['OPENAI_API_KEY'] = 'your-api-key'
os.environ['DEPLOYMENT_NAME'] = 'test-chat'
os.environ['MODEL_NAME'] = 'test-chat'
os.environ['EMBED_MODEL_NAME'] = 'test-embedding-ada'
os.environ['EMBED_DEPLOYMENT_NAME'] = 'test-embedding-ada'


def get_model(model_name):
    """
        Get the model of user choose
    Aargs
        model_name: Name of model to use. Should be one of "gpt4all", "azure-openai", "llama",
        "llama2", "chat-glm" and "chat-glm2"
    Returns:
        large language model
    Raises:
        An error is name of model be not supported.
    """
    # Initialize large language model and embedding model
    if model_name == 'gpt4all':
        llm = GPT4All(model=GPT4ALL, backend='gptj', callbacks=None, verbose=False)
    elif model_name == 'azure-openai':
        llm = AzureOpenAI(deployment_name=os.getenv('DEPLOYMENT_NAME'), model_name=os.getenv('MODEL_NAME'))
    elif model_name == 'llama':
        llm = CTransformers(model=LLAMA, model_type='llama', config=LLAMA_CONFIG)

    elif model_name == 'llama2':
        llm = CTransformers(model=LLAMA, model_type='llama', config=LLAMA_CONFIG)
    elif model_name == 'chat-glm':
        llm = ChatLLM()
        llm.load_llm(model_name=ChatGLM)

    elif model_name == 'chat-glm2':
        llm = ChatLLM()
        llm.load_llm(model_name=ChatGLM2)

    else:
        raise [ValueError('The args "model_name" must be one of "gpt4all", "azure-openai", '
                          '"llama", "llama2", "chat-glm" and "chat-glm2"')]
    return llm


def get_embeddings(model_name):
    """
        Get the embeddings of model
    Args:
        model_name: Name of model to use. Should be one of "gpt4all", "azure-openai", "llama",
        "llama2", "chat-glm" and "chat-glm2"
    Returns:
        embedding model
    """
    if model_name == 'gpt4all':
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL,
                                           model_kwargs={'device': DEVICE},
                                           cache_folder=TRANSFORMERS_PATH)
    else:
        embeddings = OpenAIEmbeddings(model="test-embedding-ada", deployment="test-embedding-ada",
                                      chunk_size=1)
    return embeddings


def get_vector(vector_db_name='chroma'):
    """
        Get the vector database
    Args:
        vector_db_name: Name of vectors database. Should be one of "chroma" and "faiss"
    Returns:
        vector database
    Raises:
        An error is name of vector database be not supported.
    """
    # Initialize database of vectors
    if vector_db_name == 'chroma':
        db = Chroma
    elif vector_db_name == 'faiss':
        db = FAISS
    else:
        raise [ValueError('The args "vector_db_name" must be one of "chroma" and "faiss"')]
    return db


class ChatLLM(LLM):
    """
    Initialize ChatLLM use LangChain base LLM class.
    """
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    model_type: str = "chat_glm"
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        """Return the type of llm."""
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        r"""Call out to chat_glm's generate method.
        Args:
            prompt: The prompt to pass into the model.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                prompt = "Once upon a time, "
                response = model(prompt, n_predict=55)
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response, self.history = self.model.chat(self.tokenizer, prompt, history=self.history)
        return response

    def load_llm(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
