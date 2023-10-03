from argparse import ArgumentParser
from vllm import LLM, SamplingParams
import time
import math

parser = ArgumentParser()
parser.add_argument("--model", default='bigscience/bloom', type=str, help="model_name")
parser.add_argument("--input_size", type=int, default=128, help="input prompt token size")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
parser.add_argument("--tensor_para_size", default=8, type=int, help="tensor parallelism")
parser.add_argument("--iters", default=5, type=int, help="number of iterations")
parser.add_argument("--greedy", action='store_true', help="greedy generation mode - temperature=0")
parser.add_argument("--print_output", action='store_true', help="print generated output text")
parser.add_argument("--test_perf", action='store_true', help="test performance to include warmup runs")
args = parser.parse_args()

max_model_len = args.max_new_tokens + args.input_size + 2
llm = LLM(args.model,
        tensor_parallel_size=args.tensor_para_size, 
        max_model_len=max_model_len, 
        disable_log_stats=False)

filename = f"/home/azureuser/GPUPower/prompt_vs_token/deepspeed_inference/prompts/bigscience/bloom/{args.input_size}.txt"
with open(filename, 'r') as f:
    prompt = f.read()
    prompts = [prompt]

if args.batch_size > len(prompts):
    prompts *= math.ceil(args.batch_size / len(prompts))

prompts = prompts[:args.batch_size]

if args.greedy:
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
else: 
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens)

# warmup
if args.test_perf:
    outputs = llm.generate(prompts, sampling_params)

for i in range(args.iters):
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print("elapsed time:", time.time() - start)

    if args.print_output:
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            #print(f"output: {output}")
            print(f"Generated text: {generated_text!r}")
