
import os
import json
from openai import OpenAI
from retry import retry
import argparse
import logging
import time
from tqdm import tqdm

# Set the OpenAI API key
client = OpenAI(api_key="your_api_key")


# Get an answer from the OpenAI-API
@retry(tries=3, delay=2, max_delay=10)
def GPT_repsonse(persona, prompt, model, temperature, max_tokens):
    messages=[{"role": "system", "content": persona["round1"]},
                {"role": "user", "content": prompt["round1"]},
                {"role": "system", "content": persona["round2"]},
                {"role": "user", "content": prompt["round2"]}
            ]
    response = client.chat.completions.create(
                                            model=model,
                                            response_format={ "type": "json_object" },
                                            messages=messages,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
    )
    response = response.choices[0].message.content
    
    return response

