from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

def vectordb(chunks):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    persist_directory = 'db'

    vectordb = Chroma.from_texts(chunks, embedding_function, persist_directory=persist_directory)

    vectordb.persist()
    return "vector db crated successfully."