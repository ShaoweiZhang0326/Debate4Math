import os
from openai import OpenAI
from tqdm import tqdm
import re


PROXY_HTTP = ""
PROXY_HTTPS = ""
os.environ["http_proxy"] = PROXY_HTTP
os.environ["https_proxy"] = PROXY_HTTPS

DEFAULT_PROMPT = (
    "hello! This is a test text."
)

def check_openai_network(model="gpt-3.5-turbo") -> bool:
    print(f"Testing connection to {model} API...")
    if "gpt" in model.lower():
        api_key = ''
        client = OpenAI(
            api_key=api_key,
        )
    elif "qwen" in model.lower():
        api_key = ''
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    try:
        # test_client = OpenAI(api_key=api_key)
        test_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say this is a test"}]
        )
        print("Connection successful!")
        return client
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False

def process_question(
    client,
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0
    ) -> tuple[str, int]:
    # context = [{"role": "user", "content": prompt}]
    context = prompt
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=context,
            temperature=temperature
        )
        answer = completion.choices[0].message.content
        total_tokens = completion.usage.total_tokens
        return answer, total_tokens
    except Exception as e:
        print(f"Error occurred during processing: {e}")
        return "", 0
