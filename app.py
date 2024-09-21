from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

def pdf_directory_loader(path):
    loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text = ""
    for doc in documents:
        text+=doc.page_content
    return text

def preprocess_text(text):
    preprocessText = text.replace("\n"," ")
    return preprocessText


def semantic_text_spliter(preprocessText,min_token=200,max_token=1000):
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, capacity=(200,1000))

    chunks = splitter.chunks(preprocessText)
    return chunks


def vectordb(chunks):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    persist_directory = 'db'

    vectordb = Chroma.from_texts(chunks, embedding_function, persist_directory=persist_directory)

    vectordb.persist()
    return "vector db crated successfully."

def similarity_search(query):
    docs = vectordb.similarity_search(query,k=5)
    for doc in docs:
        yield doc

