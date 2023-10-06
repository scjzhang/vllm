import argparse
from datetime import datetime

from vllm import EngineArgs, HYBRIDLLMEngine, SamplingParams


def main(args: argparse.Namespace):
   # Parse the CLI argument and initialize the engine.
   engine_args = EngineArgs.from_cli_args(args)
   engine = HYBRIDLLMEngine.from_engine_args(engine_args)

   # Test the following prompts.
   test_prompts = [
       ("A robot may not injure a human being",
        SamplingParams(temperature=0.0)),
       ("To be or not to be,",
        SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
       ("What is the meaning of life?",
        SamplingParams(temperature=0.8,
                       top_p=0.95,
                       frequency_penalty=0.1)),
       ("It is only with the heart that one can see rightly",
        SamplingParams(temperature=0.0)),
   ]

   # Run the engine by calling `engine.step()` manually.
   request_id = 5
   step_id = 0
   while True:
       # To test continuous batching, we add one request at each step.
       if test_prompts:
           prompt, sampling_params = test_prompts.pop(0)
           engine.add_request(str(request_id), prompt, sampling_params)
           request_id -= 1

       request_outputs = engine.step()
       step_id += 1
       print(f"[{datetime.now()}] STEP {step_id}\n")
       for request_output in request_outputs:
           if request_output.finished:
             print("\n")
             print(f"[{datetime.now()}] FINISHED:")
           print(request_output, "\n")

       if not (engine.has_unfinished_requests() or test_prompts):`
           break


if __name__ == '__main__':
   parser = argparse.ArgumentParser(
       description='Demo on using the LLMEngine class directly')
   parser = EngineArgs.add_cli_args(parser)
   args = parser.parse_args()
   main(args)
