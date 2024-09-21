from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

import os
def pdf_directory_loader(path):
    loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text = ""
    for doc in documents:
        text+=doc.page_content
    print('loading is completed...')
    return text

def preprocess_text(text):
    preprocessText = text.replace("\n"," ")
    return preprocessText


def semantic_text_spliter(preprocessText,min_token=200,max_token=1000):
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, capacity=(min_token,max_token))

    chunks = splitter.chunks(str(preprocessText))
    print('chunks are created...')
    return chunks


def vectordb(chunks):
    embedding_function = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    persist_directory = 'db'

    vectordb = Chroma.from_texts(chunks, embedding_function, persist_directory=persist_directory)

    vectordb.persist()
    return "vector db crated successfully."


def Rerieval_step(path):
    text = pdf_directory_loader(path)
    preprocessText = preprocess_text(text)

    chunks = semantic_text_spliter(preprocessText)

    return vectordb(chunks)



if __name__=="__main__":
    path = 'ENGINEERING'
    Rerieval_step(path)