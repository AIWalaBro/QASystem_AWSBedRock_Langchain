from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Bedrock
from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
import boto3
from data_ingestion.ingestion import get_vector_store, data_ingestion

# connect to client server
bedrock = boto3.client(service_name = 'bedrock-runtime')


# design prompt template
prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template= prompt_template, input_variables=['context','question']
)

# lets import the llms module
def get_llama2_llm():
    llm = Bedrock(model_id = "meta.llama2-13b-chat-v1", client = bedrock, model_kwargs = {'max_tokens':512})
    return llm

# retrieval fucntions
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vectorstore_faiss.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k": 3}
            ),
        return_source_documents = True,
        chain_type_kwargs = {"prompt": PROMPT},
       
    )
    answer = qa({'query': query})
    return answer['result']

if __name__ == "__main__":
    docs = data_ingestion()
    vectorstore_faiss = get_vector_store(docs)
    query = "when the india is born"
    llm = get_llama2_llm()
    get_response_llm(llm,vectorstore_faiss,query)
    
    # faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
    # query="What is RAG token?"
    # llm=get_llama2_llm()
    # print(get_response_llm(llm,faiss_index,query))