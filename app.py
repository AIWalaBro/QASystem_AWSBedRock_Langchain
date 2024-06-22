import os
import sys
import boto3
import streamlit as streamlit

from langchain.community.embeddings import bedrockembeddings
from langchain.llms.bedrock import bedrock

from langchain.prompts import PromptTemplates
from langchain.chains import RetrievelQA
from langchain.vectorstores import FAISS

