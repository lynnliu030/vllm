import random
import time
import string
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import random 

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", tokenizer="hf-internal-testing/llama-tokenizer")
sampling_param = SamplingParams(temperature=0, spaces_between_special_tokens=False)

def generate_random_string(token_length: int) -> str:
    # set seed 
    random.seed(0)
    
    num_special_tokens = len(tokenizer.encode("", add_special_tokens=True)) - len(tokenizer.encode("", add_special_tokens=False))
    adjusted_token_length = token_length - num_special_tokens
    if adjusted_token_length <= 0:
        raise ValueError("Token length must be greater than the number of special tokens added by the tokenizer.")

    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=1000))
    tokenized_output = tokenizer.encode(random_string, add_special_tokens=False)
    
    if len(tokenized_output) > adjusted_token_length:
        truncated_output = tokenized_output[:adjusted_token_length]
    else:
        truncated_output = tokenized_output + [tokenizer.pad_token_id] * (adjusted_token_length - len(tokenized_output))

    decoded_string = tokenizer.decode(truncated_output, skip_special_tokens=False)
    return decoded_string

def generate_unique_prefix(base_text, index):
    index_str = str(index)
    if len(index_str) > len(base_text):
        raise ValueError("Index is too large to fit into the base prefix.")

    # Replace beginning of the base_text with index_str
    return index_str + base_text[len(index_str):]

def test_prefix_caching(num_prefix, num_samples_per_prefix, prefix_length, input_length, reorder=False, prefix=True):
    tot_num_reqs = num_prefix * num_samples_per_prefix
    
    base_prefix = generate_random_string(prefix_length)
    suffix_length = input_length - prefix_length
    suffix = generate_random_string(suffix_length)
    
    tokenized_suffix_length = len(tokenizer.encode(suffix))
    assert tokenized_suffix_length == suffix_length, f"Suffix token length: {tokenized_suffix_length}, Expected suffix token length: {suffix_length}"
    
    total_time = 0
    total_output_token_length = 0
    total_input_token_length = 0

    prompt_list = []
    propmt_pos_list = []
    input_token_list = [] 
    
    # Generate unique prefixes 
    for i in range(num_prefix):
        unique_prefix = generate_unique_prefix(base_prefix, i)
        print(f"Unique prefix {i}: {unique_prefix}")
        prompt = unique_prefix + suffix

        prefix_token_length = len(tokenizer.encode(unique_prefix))
        assert prefix_token_length == prefix_length, f"Prefix token length: {prefix_token_length}, Expected prefix token length: {prefix_length}"
        
        # prefix_token_length = None 
        input_token_length = len(tokenizer.encode(prompt))
        total_input_token_length += input_token_length
        
        prompt_list.append(prompt)
        propmt_pos_list.append(prefix_token_length)
        input_token_list.append(input_token_length)
    
    # Test with prefix 1,2,3,...,N, 1,2,3,...,N, 1,2,3,...,N, ...
    if not prefix:
        print(f"Sequential, no prefix")
        prompt_inputs = prompt_list*num_samples_per_prefix
        
        st_time = time.time()
        output = llm.generate(prompt_inputs, prefix_pos=None, sampling_params=sampling_param)
        end_time = time.time()
    else: 
        if not reorder:
            print(f"Testing with prefix 1,2,3,...,N, 1,2,3,...,N, 1,2,3,...,N, ...")
            prompt_inputs = prompt_list*num_samples_per_prefix
            prompt_pos_inputs = propmt_pos_list*num_samples_per_prefix
            assert len(prompt_inputs) == len(prompt_pos_inputs) == tot_num_reqs, f"Prompt input length: {len(prompt_inputs)}, Prompt pos input length: {len(prompt_pos_inputs)}, Expected length: {tot_num_reqs}"
            
            st_time = time.time()
            output = llm.generate(prompt_inputs, prefix_pos=prompt_pos_inputs, sampling_params=sampling_param)
            end_time = time.time()
        else:
            # reorder prefix so that it is 1*num_samples_per_prefix, 2*num_samples_per_prefix, 3*num_samples_per_prefix, ...
            # Reorder the prompts
            print(f"Reordering prompts...")
            reordered_prompts = []
            reordered_prefix_pos = []
            for i in range(num_prefix):
                reordered_prompts.extend([prompt_list[i]] * num_samples_per_prefix)
                reordered_prefix_pos.extend([propmt_pos_list[i]] * num_samples_per_prefix)

            assert len(reordered_prompts) == len(reordered_prefix_pos) == tot_num_reqs, f"Reordered prompt length: {len(reordered_prompts)}, Reordered prefix pos length: {len(reordered_prefix_pos)}, Expected length: {tot_num_reqs}"
            st_time = time.time()
            output = llm.generate(reordered_prompts, prefix_pos=reordered_prefix_pos, sampling_params=sampling_param)
            end_time = time.time()
    
    i = 0
    for o in output:
        output_token_length = len(tokenizer.encode(o.outputs[0].text))
        total_output_token_length += output_token_length
        print(f"Index: {i}, Output token length: {output_token_length}, Output: {o.outputs[0].text}")
        i += 1 
    
    print(f"Total input token length: {total_input_token_length}, Total output token length: {total_output_token_length}")
    total_time += end_time - st_time
    avg_latency = total_time / tot_num_reqs
    rps = tot_num_reqs / total_time
    tps = (total_input_token_length + total_output_token_length) / total_time

    print(f"\nAverage Latency: {avg_latency:.2f} s/request")
    print(f"Throughput: {rps:.2f} requests/s, {tps:.2f} tokens/s")

# Test with a large number of unique prefixes
num_prefix = 50 # Number of unique prefix   
num_samples_per_prefix = 10 # Number of samples for each unique prefix 
tot_num_reqs = num_prefix * num_samples_per_prefix # Total number of requests
print(f"Total number of unique prefixes: {num_prefix}, Total number of requests: {tot_num_reqs}, Number of samples for each unique prefix: {num_samples_per_prefix}")

prefix_length = 400 # Number of tokens in the prefix
input_length = 405 # Number of tokens in the input NOTE: suffix needs to be greater than 3 
suffix_length = input_length - prefix_length # Number of tokens in the suffix
print(f"Prefix length: {prefix_length}, Suffix length: {suffix_length}, Input length: {input_length}")

test_prefix_caching(num_prefix, num_samples_per_prefix, prefix_length, input_length, reorder=False)