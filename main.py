import os
import sys
import boto3
import json
print("import sucessfully..!")

prompt = """
    your are an smartest assistant. and let me know what is the machine learning in smortest way
"""
bedrock = boto3.client(service_name = 'bedrock-runtime')
payload = {
    
}

body = json.dumps(payload)
model_id = "meta.llama3-70b-instruct-v1:0" 

response = bedrock.invoke_model(
    body=body,
    model_id=model_id,
    accept = 'application/json',
    content_type = 'application/json'
)

response_body = json.loads(response.get("body").read())
response_text = response_body['generation']
print(response_text)
