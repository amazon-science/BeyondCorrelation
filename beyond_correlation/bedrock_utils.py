import json
import time
import traceback
from functools import partial

import botocore.config
from boto3 import session as boto3_session


def predict_llama(prompt, temperature=1.0, repeat=5, model_id="meta.llama3-1-8b-instruct-v1:0"):
    current_session = boto3_session.Session()
    bedrock = current_session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        config=botocore.config.Config(
            read_timeout=120,  # corresponds to inference time limit set for Bedrock
            connect_timeout=120,
            retries={
                "max_attempts": 5,
            },
        ),
    )
    api_template = {
        "modelId": model_id,
        "body": "",
    }

    llama_wrapper = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|>\n\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    prompt_text = llama_wrapper.format(prompt=prompt)
    body = {"max_gen_len": 64, "temperature": temperature, "top_p": 0.8, "prompt": prompt_text}
    api_template["body"] = json.dumps(body)

    all_results = []
    for i in range(repeat):
        for retry in range(10):
            try:
                response = bedrock.invoke_model_with_response_stream(**api_template)
                response_text = ""
                for event in response["body"]:
                    chunk = json.loads(event["chunk"]["bytes"])
                    if "generation" in chunk:
                        response_text += chunk["generation"]
                all_results.append(response_text.strip())
                break
            except:
                traceback.print_exc()
                time.sleep(5)
    return all_results


def predict_claude(prompt, temperature=1.0, repeat=5, model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
    current_session = boto3_session.Session()
    bedrock = current_session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        config=botocore.config.Config(
            read_timeout=120,  # corresponds to inference time limit set for Bedrock
            connect_timeout=120,
            retries={
                "max_attempts": 5,
            },
        ),
    )
    api_template = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "*/*",
        "body": "",
    }

    body = {
        "max_tokens": 64,
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 10,
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }

    api_template["body"] = json.dumps(body)

    all_results = []
    for i in range(repeat):
        for retry in range(10):
            try:
                response = bedrock.invoke_model(**api_template)
                response_body = json.loads(response.get("body").read())
                all_results.append(response_body["content"][0]["text"].strip())
                break
            except:
                traceback.print_exc()
                time.sleep(5)
    return all_results


def predict_mistral(prompt, temperature=1.0, repeat=5, model_id="mistral.mistral-7b-instruct-v0:2"):
    current_session = boto3_session.Session()
    bedrock = current_session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        config=botocore.config.Config(
            read_timeout=120,  # corresponds to inference time limit set for Bedrock
            connect_timeout=120,
            retries={
                "max_attempts": 5,
            },
        ),
    )
    api_template = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "*/*",
        "body": "",
    }

    mistral_wrapper = "<s>[INST]{prompt}[/INST]"
    prompt_text = mistral_wrapper.format(prompt=prompt)
    body = {"max_tokens": 64, "temperature": temperature, "top_p": 0.8, "top_k": 50, "prompt": prompt_text}
    api_template["body"] = json.dumps(body)

    all_results = []
    for i in range(repeat):
        for k in range(10):
            try:
                response = bedrock.invoke_model(**api_template)
                response_body = json.loads(response.get("body").read())
                all_results.append(response_body["outputs"][0]["text"].strip())
                break
            except:
                traceback.print_exc()
                time.sleep(5)
    return all_results


llm_configs = {
    "sonnet_3": partial(predict_claude, model_id="anthropic.claude-3-sonnet-20240229-v1:0"),
    "llama_3_70B": partial(predict_llama, model_id="meta.llama3-70b-instruct-v1:0"),
    "mixtral_8x7B": partial(predict_mistral, model_id="mistral.mixtral-8x7b-instruct-v0:1"),
}
