from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


def load_documents(directory_path, pattern="**/*.txt"):
    loader = DirectoryLoader(directory_path, glob=pattern, loader_cls=TextLoader, show_progress=True)
    return loader.load()


def split_documents(documents, chunk_size=500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    return text_splitter.split_documents(documents)


def store_documents(documents, embeddings_model):
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings_model)
    return vectorstore.as_retriever(search_kwargs={'k': 2})


def prepare_document_retriever(data_directory, azure_endpoint="azure_endpoint", api_version="2024-02-01"):
    docs = load_documents(data_directory)
    splits = split_documents(docs)
    embeddings_model = AzureOpenAIEmbeddings(azure_endpoint=azure_endpoint, openai_api_version=api_version)
    retriever = store_documents(splits, embeddings_model)
    return retriever
