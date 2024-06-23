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
    vector_store_faiss=FAISS.from_documents(docs,bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss
    

if __name__ == '__main__':
    docs = data_ingestion()
    get_vector_store(docs)


# def get_vector_store(docs):
#     if not docs:
#         print("No documents found after splitting.")
#         return
#     print(f"Number of documents to embed: {len(docs)}")

#     # Create a 'vector_store' directory at the root level of the project
#     project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     vector_store_dir = os.path.join(project_root_dir, 'vector_store')
#     if not os.path.exists(vector_store_dir):
#         os.makedirs(vector_store_dir)

#     # Save the FAISS index in the 'vector_store' directory
#     vector_store_path = os.path.join(vector_store_dir, 'faiss_index')
#     try:
#         vector_store_faiss = FAISS.from_documents(docs, bedrock_embeddings)
#         vector_store_faiss.save_local(vector_store_path)
#         print(f"FAISS index saved at: {vector_store_path}")
#         return vector_store_faiss
#     except Exception as e:
#         print(f"Error while creating FAISS vector store: {e}")
