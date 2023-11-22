import argparse
import random
import time
from typing import List, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

prefix = f"""
Answer whether {{reviewText}} suggest good quality as indicated in the product description {{description}},
If yes, just answer a single word "yes", if no, answer a single word "no". Don't answer anything else.

Example: 
If description = "Loud 'N Clear Personal Sound Amplifier allows you to turn up the
volume on what people around you are saying, listen at the level you want without 
disturbing others, hear a pin drop from across the room." and 
reviewText = "very quite", then the good quality matches, return "yes this is good". 

If reviewText = "bad quality, load",
then answer "no, this is not good", the quality does not match. Here is reviewText and description:
"""*5
req = "reviewText = \"very quite\", description = \"Loud 'N Clear Personal Sound Amplifier allows you to turn up the volume on what people around you are saying, listen at the level you want without disturbing others, hear a pin drop from across the room.\""
prompt = prefix+req
prefix_token_length = len(tokenizer.encode(prefix))
input_token_length = len(tokenizer.encode(prefix+req))
suffix_token_length = input_token_length - prefix_token_length
print(f"Prefix token length: {prefix_token_length}")
print(f"Input token length: {input_token_length}")

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", tokenizer="hf-internal-testing/llama-tokenizer")
sampling_param = SamplingParams(temperature=0, spaces_between_special_tokens=False)
output = llm.generate([prompt], sampling_params=sampling_param)

num_req = 10

# prefix_pos = None 
prefix_pos = [prefix_token_length]*num_req
print(f"Prefix position: {prefix_pos}")

tot_time = 0 
output_token_length = 0

st_time = time.time()
output = llm.generate([prompt]*num_req, prefix_pos=prefix_pos, sampling_params=sampling_param)
end_time = time.time()
tot_time += end_time - st_time

for out in output:
    output_token_length += len(tokenizer.encode(out.outputs[0].text))

if prefix_pos is None:
    input_token_length = input_token_length*num_req
else: 
    input_token_length = suffix_token_length*num_req
    
print(f"Output token length: {output_token_length}")

rps = num_req/ tot_time
tps = (input_token_length + output_token_length) / tot_time
avg_latency = tot_time / num_req

print(f"Throughput: {rps:.2f} requests/s, {tps:.2f} tokens/s")
print(f"Avg Latency: {avg_latency:.2f} s/request")

prefix2 = f"""
Answer whether {{reviewText}} suggest good quality as indicated in the product description {{description}},
If yes, just answer a single word "yes", if no, answer a single word "no". Don't answer anything else.

Example: 
If description = "Loud 'N Clear Personal Sound Amplifier allows you to turn up the
volume on what people around you are saying, listen at the level you want without 
disturbing others, hear a pin drop from across the room." answer yes, no, or this is soso.
"""*5
req2 = req 
prompt2 = prefix2+req2
prefix_token_length2 = len(tokenizer.encode(prefix2))
input_token_length2 = len(tokenizer.encode(prefix2+req2))
suffix_token_length2 = input_token_length2 - prefix_token_length2
print(f"Prefix2 token length: {prefix_token_length2}")
print(f"Input2 token length: {input_token_length2}")

# prefix_pos = None 
prefix_pos = [prefix_token_length]*(num_req//2) + [prefix_token_length2]*(num_req//2)

second_prompt = [prompt]*(num_req // 2) + [prompt2]* (num_req // 2)
tot_time = 0 
output_token_length = 0

print(f"Second prompt: {second_prompt}, prefix_pos: {prefix_pos}")
st_time = time.time()
output = llm.generate(second_prompt, prefix_pos=prefix_pos, sampling_params=sampling_param)
end_time = time.time()
tot_time += end_time - st_time
print(f"Output: {output}")
for out in output:
    output_token_length += len(tokenizer.encode(out.outputs[0].text))

if prefix_pos is None:
    input_token_length = input_token_length*num_req
else: 
    input_token_length = suffix_token_length*num_req
    
print(f"Output token length: {output_token_length}")

rps = num_req/ tot_time
tps = (input_token_length + output_token_length) / tot_time
avg_latency = tot_time / num_req

print(f"Throughput: {rps:.2f} requests/s, {tps:.2f} tokens/s")
print(f"Avg Latency: {avg_latency:.2f} s/request")