from vllm import LLM, SamplingParams
import time
import math

llm = LLM("bigscience/bloom", tensor_parallel_size=8, max_model_len=4800, max_num_batched_tokens=4300, disable_log_stats=False)

filename = f"/home/azureuser/GPUPower/prompt_vs_token/deepspeed_inference/prompts/bigscience/bloom/512.txt"
with open(filename, 'r') as f:
    prompt = f.read()
    prompts = [prompt]

batch_size = 8
if batch_size > len(prompts):
    prompts *= math.ceil(batch_size / len(prompts))

prompts = prompts[:batch_size]

outputs = llm.generate(prompts)
iters = 5
sampling_params = SamplingParams(temperature=0, top_k=1, max_tokens=128)
for i in range(iters):
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print("elapsed time:", time.time() - start)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        #print(f"output: {output}")
        print(f"Generated text: {generated_text!r}")
