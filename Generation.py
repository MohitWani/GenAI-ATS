from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def retrieval():
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    

    # Create a Hugging Face Pipeline LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain


def similarity_search(query):
    print("Retrieving similar documents...")

    embedding_function = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    persist_directory = "db"

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    docs = vectordb.similarity_search(query,k=5)
    for doc in docs:
        yield doc

def response(docs,chain,query):
    response = chain(
        {"input_documents":docs, "question": query}
        , return_only_outputs=True)
    
    return response['output_text']

if __name__=="__main__":

    query = "Which are skills are mention in documents"
    for i in similarity_search(query):
        print(i)
        print("\n")