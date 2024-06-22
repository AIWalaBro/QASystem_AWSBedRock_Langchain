from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock

import json
import sys
import os
import boto3

## Bedrock client
bedrock = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client = bedrock)

def data_ingestion():
    loader = PyPDFDirectoryLoader("D:\\GenerativeAI_Project_4\\aws\\QA_System_With_AWSBedrock_and_Langchain\\RAG_System_AWSBedRock_ECR_DockerLangchain\\data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap = 1000)
    docs = text_splitter.split_documents(documents)
    # print('your document chunk succesfully created!')
    return docs
    

def get_vector_store(docs):
    # print('faiss indexing local vector store creation started...')
    vector_store_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")
    # print('faiss indexing local vector store succesfully created!')
    return vector_store_faiss
    
if __name__ == '__main__':
    docs = data_ingestion()
    get_vector_store(docs)



#------------------------------------------------------- for debugging ----------------------------------------------------------------------

'''
you can use the following functions to create a data ingestion and vecetore store creation,
it just an debugging function where you can each steps where you code getting error or runnin
sucessfully.
'''

# def data_ingestion():
#     loader = PyPDFDirectoryLoader("D:\\GenerativeAI_Project_4\\aws\\QA_System_With_AWSBedrock_and_Langchain\\RAG_System_AWSBedRock_ECR_DockerLangchain\\data")
#     documents = loader.load()
#     if not documents:
#         print("No documents loaded from the directory.")
#         return []
#     print(f"Number of documents loaded: {len(documents)}")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
#     docs = text_splitter.split_documents(documents)
#     if not docs:
#         print("No documents found after splitting.")
#     else:
#         print(f"Number of documents after splitting: {len(docs)}")
#         print(f"Sample document: {docs[0]}")
#     return docs


# def get_vector_store(docs):
#     if not docs:
#         print("No documents found after splitting.")
#         return
#     print(f"Number of documents to embed: {len(docs)}")
#     try:
#         vector_store_faiss = FAISS.from_documents(docs, bedrock_embeddings)
#         vector_store_faiss.save_local("faiss_index")
#     except Exception as e:
#         print(f"Error while creating FAISS vector store: {e}")

