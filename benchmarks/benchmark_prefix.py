import argparse
import random
import time
from typing import List, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", tokenizer="hf-internal-testing/llama-tokenizer")
sampling_param = SamplingParams(temperature=0, spaces_between_special_tokens=False)

def generate_unique_prefix(base_text, index):
    return base_text.replace('{{index}}', str(index))

def test_prefix_caching(num_tests):
    base_prefix = """
    Answer whether {{reviewText}} suggest good quality as indicated in the product description {{description}},
    If yes, just answer a single word "yes", if no, answer a single word "no". Don't answer anything else.

    Example: 
    If description = "Loud 'N Clear Personal Sound Amplifier allows you to turn up the
    volume on what people around you are saying, listen at the level you want without 
    disturbing others, hear a pin drop from across the room." and 
    reviewText = "very quite", then the good quality matches, return "yes this is good". 
    Here is reviewText and description for request {{index}}:
    """*10

    req = "reviewText = \"very quite\", description = \"Loud 'N Clear Personal Sound Amplifier allows you to turn up the volume on what people around you are saying, listen at the level you want without disturbing others, hear a pin drop from across the room.\""
    
    total_time = 0
    total_output_token_length = 0

    prompt_list = []
    propmt_pos_list = []
    input_token_list = [] 
    
    for i in range(num_tests):
        if len(prompt_list) < 10: 
            unique_prefix = generate_unique_prefix(base_prefix, i)
            prompt = unique_prefix + req

            prefix_token_length = len(tokenizer.encode(unique_prefix))
            # prefix_token_length = None 
            input_token_length = len(tokenizer.encode(prompt))
            print(f"Prefix token length: {prefix_token_length}, Input token length: {input_token_length}")
            
            prompt_list.append(prompt)
            propmt_pos_list.append(prefix_token_length)
            input_token_list.append(input_token_length)
        else:
            prompt = prompt_list[i % 10]
            prefix_token_length = propmt_pos_list[i % 10]
            input_token_length = input_token_list[i % 10]
        
        if prefix_token_length != None:
            prefix_token_length = [prefix_token_length]
            
        st_time = time.time()
        output = llm.generate([prompt], prefix_pos=prefix_token_length, sampling_params=sampling_param)
        end_time = time.time()

        output_token_length = len(tokenizer.encode(output[0].outputs[0].text))
        total_time += end_time - st_time
        total_output_token_length += output_token_length

        print(f"Test {i+1}/{num_tests}")
        print(f"Generated Output: {output}")
        print(f"Latency: {end_time - st_time:.2f} s")

    avg_latency = total_time / num_tests
    rps = num_tests / total_time
    tps = (input_token_length + total_output_token_length) / total_time

    print(f"\nAverage Latency: {avg_latency:.2f} s/request")
    print(f"Throughput: {rps:.2f} requests/s, {tps:.2f} tokens/s")

# Test with a large number of unique prefixes
num_tests = 40 # Adjust this number based on your testing needs
test_prefix_caching(num_tests)
