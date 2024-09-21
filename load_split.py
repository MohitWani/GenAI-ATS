from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

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