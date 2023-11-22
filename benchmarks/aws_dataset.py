import argparse
import random
import time
from typing import List, Tuple

from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, LongType, ArrayType, MapType
from pyspark.sql.functions import col
from pyspark.sql import functions as F

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("AmazonBeautyBenchmark").getOrCreate()


class LLMWrapper:
    def __init__(self, model: str, tokenizer: str):
        self.sampling_params = SamplingParams(temperature=0, spaces_between_special_tokens=False, max_tokens=1)
        self.llm = LLM(model=model, tokenizer=tokenizer)

    def run_llm(self, requests: List[Tuple[str, str]], n: int) -> float:
        # Prefix length = 658
        query = f"""
        Answer whether {{reviewText}} suggest good quality for the product, and analayze the sentiment, return a single word "good", "bad", or "ok".

        Example: 
        If reviewText = "HEY!! I am an Aqua Velva Man and absolutely love this stuff, been using it for over 50 years. This is a true after shave lotion classic. Not quite sure how many women that have been attracted to me because of Aqua Velva,  I do know for sure that it's just to many to count. Ha.  Not sure how long this has been around but the Williams Company ran a paper advertisement, taken from a 1949 magazine, which features Ralph Bellamy of Detective Story and Ezio Pinza of South Pacific for Aqua Velva After Shave Lotion. I'm sure you all remember Ralph Bellamy and Ezio Pinza from the 40's right.\n\nThere slogan was \" There is nothing like an AQUA VELVA MAN \" and it seemed to work because this was a hot item when I was a young man. It's just amazing that this is still around when the vast majority of after shave lotions from that time period have disappeared over the years.\n\nThe only negative about Aqua Velva is it does not last very long. Usually in less than an hour the aroma has dissipated. The more expensive after shave lotions and perfumes have Ambergris in them which is quite expensive and comes from a sperm whale this enables the aroma to linger much longer.\n\nAt any rate this is a great classic after shave with a wonderful refreshing odor, give it a try you might just like it.\n\nNote :::; Some reviewers mention this 7oz size now comes in a plastic container, I just bought one in Wal-Mart and it was glass.", then good quality matches, return "good". 

        If reviewText = "Mindee Urbis", "reviewText": "I'm giving this three stars because it does work but the results do not justify the price.\n\nI've been using this for over two months. Before I started I couldn't even get my lashes on a curler. They were that short. Within two and a half weeks I could see a difference. Within a month I could get my lashes on a curler.\n\nI've used this as directed.  I did see improvement but I think there is a natural plateau to the lenght and amount of eyelashes that one's DNA will allow. I've hit mine.  Overall a good product but I can't see spending the money on it again (I got mine for 60% off and still can't justify it). Then return "ok".
        
        If reviewText = "bad quality, smell so bad, will not buy it ever again", then answer "bad". 
        """

        # Prepare the prompts
        prompts = [f"{query}. What's your answer for reviewText: {review}?" for review, _ in requests]

        # tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        # print(f"Prefix length: {len(tokenizer.encode(query))})")

        # # Average number of suffix tokens = 64
        # avg_suffix_tokens = sum(len(tokenizer.encode(f"What's your answer for reviewText: {review}?")) for review, _ in requests)/len(requests)
        # print(f"Average number of suffix tokens: {avg_suffix_tokens}")

        start = time.perf_counter()
        outputs = self.llm.generate(prompts, self.sampling_params)
        end = time.perf_counter()

        outputs = [output.outputs[0].text for output in outputs]

        return end - start, prompts, outputs


def sample_requests_from_sparkdf(joined_df, num_samples: int) -> List[Tuple[str, str]]:
    # Sample rows from the Spark DataFrame
    sampled_df = joined_df.sample(False, num_samples / joined_df.count()).limit(num_samples).collect()

    # Extract reviewText and description pairs
    samples = [(row["reviewText"], row["description"]) for row in sampled_df]
    return samples


def read_dataset(data_path: str = "../nb/Amazon_beauty.json", metadata_path: str = "../nb/Amazon_meta_beauty.json"):
    schema = StructType(
        [
            StructField("overall", DoubleType(), True),
            StructField("verified", BooleanType(), True),
            StructField("reviewTime", StringType(), True),
            StructField("reviewerID", StringType(), True),
            StructField("asin", StringType(), True),
            StructField("style", StructType([StructField("Size:", StringType(), True), StructField("Color:", StringType(), True)]), True),
            StructField("reviewerName", StringType(), True),
            StructField("reviewText", StringType(), True),
            StructField("summary", StringType(), True),
            StructField("unixReviewTime", LongType(), True),
        ]
    )
    df = spark.read.json(data_path, schema=schema)

    schema = StructType(
        [
            StructField("asin", StringType(), True),
            StructField("title", StringType(), True),
            StructField("feature", ArrayType(StringType(), True), True),
            StructField("description", StringType(), True),
            StructField("price", DoubleType(), True),
            StructField("imageURL", StringType(), True),
            StructField("imageURLHighRes", StringType(), True),
            StructField("also_buy", ArrayType(StringType(), True), True),
            StructField("also_viewed", ArrayType(StringType(), True), True),
            StructField("salesRank", MapType(StringType(), LongType(), True), True),
            StructField("brand", StringType(), True),
            StructField("categories", ArrayType(ArrayType(StringType()), True), True),
            StructField("tech1", ArrayType(StringType(), True), True),
            StructField("tech2", ArrayType(StringType(), True), True),
            StructField("similar", ArrayType(MapType(StringType(), StringType()), True), True),
        ]
    )

    meta_df = spark.read.json(metadata_path, schema=schema)
    joined_df = df.join(meta_df, on="asin", how="inner")

    # Filter out rows where reviewText or description contain None values
    joined_df = joined_df.filter((col("reviewText").isNotNull()) & (col("description").isNotNull()))
    joined_df.persist()
    joined_df.createOrReplaceTempView("joined_df")

    return joined_df


def main(args: argparse.Namespace, llm_wrapper: LLMWrapper):
    # Sample the requests
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    joined_df = read_dataset()
    requests = sample_requests_from_sparkdf(joined_df, args.num_prompts)

    elapsed_time, prompts, outputs = llm_wrapper.run_llm(requests, args.n)

    total_input_tokens, total_output_tokens = 0, 0
    for prompt in prompts:
        try:
            total_input_tokens += len(tokenizer.encode(prompt))
        except Exception as e:
            print(f"Error encoding prompt: {prompt}. Error: {e}")

    print(f"Total number of input tokens: {total_input_tokens}")
    for output in outputs:
        try:
            print(f"Output: {output}")
            total_output_tokens += len(tokenizer.encode(output))
        except Exception as e:
            print(f"Error encoding output: {output}. Error: {e}")

    print(f"Total number of output tokens: {total_output_tokens}")
    total_num_tokens = total_input_tokens + total_output_tokens
    print(f"Total number of tokens: {total_num_tokens}")

    print(f"Elapsed time: {elapsed_time}")
    print(f"Number of requests: {len(requests)}")
    # Throughput
    rps = len(requests) / elapsed_time
    tps = total_num_tokens / elapsed_time
    # Latency
    avg_latency = elapsed_time / len(requests)

    print(f"Throughput: {rps:.2f} requests/s, " f"{tps:.2f} tokens/s")

    print(f"Avg Latency: {avg_latency:.2f} s/request")
    return rps, tps, avg_latency


def benchmark_samples(samples: List[int]):
    results = {}
    # Create a single instance of LLMWrapper
    llm_wrapper = LLMWrapper(model="meta-llama/Llama-2-7b-chat-hf", tokenizer="hf-internal-testing/llama-tokenizer")

    for num_samples in samples:
        print(f"Running benchmark with {num_samples} samples...")
        args = argparse.Namespace(
            model="meta-llama/Llama-2-7b-chat-hf", tokenizer="hf-internal-testing/llama-tokenizer", n=1, num_prompts=num_samples
        )
        rps, tps, avg_latency = main(args, llm_wrapper)
        results[num_samples] = (rps, tps, avg_latency)
    return results


def plot_results(results, save_path):
    sample_sizes = list(results.keys())
    rps_values = [results[size][0] for size in sample_sizes]
    tps_values = [results[size][1] for size in sample_sizes]
    latencies = [results[size][2] for size in sample_sizes]

    plt.figure(figsize=(18, 6))

    # Plotting Requests/s Throughput
    plt.subplot(1, 3, 1)
    plt.plot(sample_sizes, rps_values, "-o", label="Requests/s")
    plt.xlabel("Sample Size")
    plt.ylabel("Throughput (Requests/s)")
    plt.title("Requests/s Throughput vs. Sample Size")
    plt.legend()

    # Plotting Tokens/s Throughput
    plt.subplot(1, 3, 2)
    plt.plot(sample_sizes, tps_values, "-o", label="Tokens/s", color="green")
    plt.xlabel("Sample Size")
    plt.ylabel("Throughput (Tokens/s)")
    plt.title("Tokens/s Throughput vs. Sample Size")
    plt.legend()

    # Plotting Average Latency
    plt.subplot(1, 3, 3)
    plt.plot(sample_sizes, latencies, "-o", color="red")
    plt.xlabel("Sample Size")
    plt.ylabel("Average Latency (s/request)")
    plt.title("Average Latency vs. Sample Size")

    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    samples = [5000]  # Example sample sizes
    results = benchmark_samples(samples)
    plot_results(results, save_path="my_plot.png")
