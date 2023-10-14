import argparse
import os
import re
from datetime import datetime

from vllm import EngineArgs, LLMEngine, SamplingParams

data_folder_name = "/data/prompts_qa"

def get_prompts():
    prompts = []
    prompt_ids = []
    file_name_pattern = r"qa_id_([0-9a-f]+)_words_([0-9]+)\.txt"
    sampling_param = SamplingParams(n=2, best_of=5, temperature=0.8, top_p=0.95, frequency_penalty=0.1)
    for file_name in os.listdir(data_folder_name):
        match = re.search(file_name_pattern, file_name)
        if match:
            prompt_id = match.group(1)
            print(f"{prompt_id}: {file_name}")
            with open(f"{data_folder_name}/{file_name}", "r", encoding="utf-8") as input_prompt_file:
                prompt = input_prompt_file.read()
                prompt_ids.append(prompt_id)
                prompts.append((prompt, sampling_param))
            # DEBUG just take one
            #break
    return prompts, prompt_ids


def main(args: argparse.Namespace):
    t0 = datetime.now()

    # Test prompts from files.
    test_prompts, test_prompt_ids = get_prompts()

    # Parse the CLI argument and initialize the engine.
    # --model bigscience/bloom
    # --tensor-parallel-size 8
    # --compress-delta 16
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)

    t1 = datetime.now()

    # Run the engine by calling `engine.step()` manually.
    request_id = 0
    step_id = 0
    while True:
        # To test continuous batching, we add one request at each step.
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs = engine.step()
        step_id += 1
        print(f"[{datetime.now()}] STEP {step_id}\n")
        for request_output in request_outputs:
            if request_output.finished:
                #print(f"[{datetime.now()}] Prompt: {request_output.prompt}")
                req_id = int(request_output.request_id)
                prompt_id = test_prompt_ids[req_id]
                print(f"[{datetime.now()}] Output {req_id} {prompt_id}: {request_output.outputs[0].text}")
                print()

                # Output answer to measure quality with quality score
                if False:
                    model_name = engine_args.model.replace('/', '_')
                    with open(f"{data_folder_name}/output_{prompt_id}_{engine_args.compress_delta}_{model_name}.txt", "w") as output_file:
                        output_file.write(request_output.outputs[0].text)

        if not (engine.has_unfinished_requests() or test_prompts):
            break

    t2 = datetime.now()

    print(f"[{datetime.now()}] Model loading: {1000.0 * (t1 - t0).total_seconds():.2f}ms")
    print(f"[{datetime.now()}] Inference: {1000.0 * (t2 - t1).total_seconds():.2f}ms")
    print(f"[{datetime.now()}] Total time: {1000.0 * (t2 - t0).total_seconds():.2f}ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--compress-delta",
        type=int,
        default=0,
        help="delta for the compression")
    args = parser.parse_args()
    main(args)
