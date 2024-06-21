import os
import sys
import boto3
import json
print("import sucessfully..!")

try:

    prompt = """
        your are an smartest assistant and let me know what is the machine learning in smartest way
    """
    bedrock = boto3.client(service_name = 'bedrock-runtime')
    payload = {
        "prompt": "[INST]"+prompt+"[/INST]",
        "max_gen_len":512,
        "temperature":0.3,
        "top_p":0.9   
    }

    body = json.dumps(payload)
    # model_id = "meta.llama3-70b-instruct-v1:0" 
    model_id = "meta.llama2-13b-chat-v1"

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept = 'application/json',
        contentType = 'application/json'
    )
    print('response generated sucessfully')

    response_body = json.loads(response.get("body").read())
    response_text = response_body['generation']
    print(response_text)
    
except Exception as e:
    print(e)
    print('error occured')
    # sys.exit(1)
