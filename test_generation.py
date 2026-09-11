import vllm

if __name__ == '__main__':
    llm = vllm.LLM("Qwen/Qwen3-0.6B-Base", tensor_parallel_size=1, gpu_memory_utilization=0.85)
    params = vllm.SamplingParams(temperature=1, max_tokens=50)
    gen = llm.generate("The capital of France is", sampling_params=params)
    print(gen[0].outputs[0].text)