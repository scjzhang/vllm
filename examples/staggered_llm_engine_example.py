import argparse

from vllm import EngineArgs, LLMEngine, SamplingParams


def main(args: argparse.Namespace):
    # Parse the CLI argument and initialize the engine.
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)

    # Test the following prompts.
    test_prompts = [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8)),
        ("What is the meaning of life?",
         SamplingParams(temperature=0.8)),
        ("It is only with the heart that one can see rightly",
         SamplingParams(temperature=0.0)),
        ("How is San Diego",
         SamplingParams(temperature=0.0)),
    ]

    # Run the engine by calling `engine.step()` manually.
    request_id = 0
    batch_id = 0
    while True:
        # To test continuous batching, we add one request at each step.
        if batch_id == 2:
            test_prompts = [
                ("Generate a list of ten titles for my autobiography",
                SamplingParams(temperature=0.0)),
                ("The book is about my journey as an adventurer who has",
                SamplingParams(temperature=0.0)),
                ("Antibiotics are a type of medication used to treat",
                SamplingParams(temperature=0.0)),
                ("Antibiotics are usually taken orally in the form of pills, capsules, or ",
                SamplingParams(temperature=0.0)),
            ]
        while test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs = engine.step()
        print(batch_id, len(request_outputs))
        batch_id += 1
        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)
                print("FINISHED:", request_output.prompt)
            print("ID: ",request_output.request_id, ": ", request_output.outputs)

        if not (engine.has_unfinished_requests() or test_prompts):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
