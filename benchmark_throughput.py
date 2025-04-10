import os
import json
import random
import argparse
import time
import subprocess
from multiprocessing import Process, Value
import numpy as np
from collections import defaultdict
import subprocess
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Memory tracking utilities
def get_gpu_memory():
    """Get the GPU memory usage using nvidia-smi command."""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        ).strip()
        
        if not result:
            return 0
        
        # Parse memory usage for each GPU
        return sum(int(x) for x in result.split('\n'))
    
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return 0

def memory_monitor(peak_memory_mb: Value, stop_flag: Value, interval=0.1):
    """Continuously check GPU memory and update peak usage."""
    while not stop_flag.value:
        current = get_gpu_memory()
        with peak_memory_mb.get_lock():
            if current > peak_memory_mb.value:
                peak_memory_mb.value = current
        time.sleep(interval)

def generate_synthetic_prompt(target_length, tokenizer):
    """Generate a synthetic prompt with approximately the target number of tokens."""
    # Base text that can be repeated to achieve desired length
    base_text = "This is a benchmark test for language models. We are testing memory usage and generation speed with various input lengths. "
    
    # Estimate tokens per base_text
    tokens_per_base_text = len(tokenizer.encode(base_text))
    
    # Calculate repetitions needed (with some margin for special tokens)
    repetitions = max(1, int(target_length / tokens_per_base_text * 0.95))
    
    # Create the synthetic prompt
    synthetic_prompt = base_text * repetitions
    
    # Add some variation to avoid model optimization for repetitive content
    synthetic_prompt += f" The current test is for approximately {target_length} tokens. Please continue this text with relevant information about language model benchmarking."
    
    # Ensure we're close to the target length
    encoded = tokenizer.encode(synthetic_prompt)
    
    # Trim or extend as needed to get closer to target
    while len(encoded) < target_length * 0.95:
        synthetic_prompt += " " + base_text
        encoded = tokenizer.encode(synthetic_prompt)
    
    while len(encoded) > target_length * 1.05:
        synthetic_prompt = synthetic_prompt[:int(len(synthetic_prompt) * 0.95)]
        encoded = tokenizer.encode(synthetic_prompt)
    
    return synthetic_prompt, len(encoded)

def format_for_model(prompt, model_name):
    """Format the prompt according to model's expected chat format if needed."""
    if "llama" in model_name.lower():
        return f"[INST] {prompt} [/INST]"
    else:
        # For other models, use their specific formats if needed
        return prompt

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def main(args):
    # Start monitor process
    peak_memory_mb = Value('i', 0)
    stop_flag = Value('b', False)
    monitor = Process(target=memory_monitor, args=(peak_memory_mb, stop_flag))
    monitor.start()
    
    # Set up the model and tokenizer
    print(f"Loading model from {args.model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        padding_side="left",
        cache_dir="/workspace"
    )
    
    from pyramidkv.monkeypatch import replace_llama
    replace_llama(args.method.lower())
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Create output directory
    model_name = args.model_path.split("/")[-1]
    output_dir = os.path.join(args.save_dir, f"{model_name}_{args.method}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define token length scenarios to test
    token_lengths = [6000]
    if args.custom_lengths:
        token_lengths = [int(x) for x in args.custom_lengths.split(',')]
    
    # Filter lengths that exceed model's context window
    model_max_length = args.model_max_length
    valid_token_lengths = [l for l in token_lengths if l <= model_max_length]
    
    if len(valid_token_lengths) < len(token_lengths):
        skipped = set(token_lengths) - set(valid_token_lengths)
        print(f"Skipping token lengths {skipped} as they exceed model's max context length of {model_max_length}")
    
    # Dictionary to store benchmark results
    benchmark_results = {
        "model": args.model_path,
        "method": args.method,
        "runs_per_length": args.runs_per_length,
        "output_tokens": args.output_tokens,
        "results": {}
    }
    
    # Load model once before testing all lengths
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=args.attn_implementation,
        cache_dir="/workspace"
    )

    quant_cache_config={
        "nbits": args.nbits,
        "backend": "HQQ",
        "device": "cuda",
        "residual_length": args.output_tokens,
        "axis_key": 1,
        "q_group_size": 64
    }

    cache_config={"backend": "quanto", "nbits": args.nbits}
    
    model.eval()
    
    # Test each token length
    for token_length in valid_token_lengths:
        print(f"\n{'='*40}")
        print(f"Testing with input length: {token_length} tokens")
        print(f"{'='*40}")
        
        # Reset length_results for this token length
        length_results = {
            "prompts": [],
            "model_memory_mb": [],
            "total_memory_mb": [],
            "total_kv_cache_mb": [],
            "prefilled_kv_cache_mb": [],
            "prefill_time_sec": [],
            "decode_time_sec": [],
            "tokens_per_sec": [],
            "actual_input_lengths": [],
            "output_lengths": []
        }
        
        # Run multiple tests for this token length for more reliable results
        for run in range(args.runs_per_length):
            print(f"\nRun {run+1}/{args.runs_per_length} for {token_length} tokens")
            
            # Generate synthetic prompt for this length
            prompt, actual_length = generate_synthetic_prompt(token_length, tokenizer)
            formatted_prompt = format_for_model(prompt, args.model_path)
            
            # Store the generated prompt for reference
            length_results["prompts"].append(prompt[:200] + "..." if len(prompt) > 200 else prompt)
            length_results["actual_input_lengths"].append(actual_length)
            
            # Clear cache between runs instead of reloading model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Make sure KV cache is cleared
            if hasattr(model, 'past_key_values') and model.past_key_values is not None:
                model.past_key_values = None
                
            print("Preparing for next run...")
            
            if args.max_capacity_prompts != -1:
                max_capacity_prompts = args.max_capacity_prompts
            
            if args.method != "FullKV":
                if args.method.lower() in ["snapkv","pyramidkv","h2o","cam", "l2norm", "adakv", "headkv", "think"]:
                    window_sizes = 8
                elif args.method.lower() in ["streamingllm"]:
                    window_sizes = max_capacity_prompts - 4

                if args.method.lower() =='headkv':
                    with open(args.head_path, 'r') as file:
                        head_list = json.loads(file.readline())
                    head_score_list = [np.mean(l[1]) for l in head_list.items()]
                    head_score_list = torch.tensor(head_score_list / sum(head_score_list))
                    total_attention = head_score_list.reshape(model.config.num_hidden_layers, model.config.num_attention_heads)
                    total_pool_capacity = (args.max_capacity_prompts // args.head_beta) * model.config.num_hidden_layers * model.config.num_attention_heads
                    min_num = (args.max_capacity_prompts - args.max_capacity_prompts // args.head_beta)
                    head_capacity = torch.round(total_attention * total_pool_capacity + min_num).int()
                    model.model.config.head_capacity = head_capacity    

                kernel_sizes = 7
                pooling = "maxpool"
                ratio = args.pruning_ratio
                recent_size = args.recent_size

                layers = len(model.model.layers)
                # check if window_sizes is a list
                if not isinstance(window_sizes, list):
                    window_sizes = [window_sizes] * layers
                if not isinstance(max_capacity_prompts, list):
                    max_capacity_prompts = [max_capacity_prompts] * layers
                if not isinstance(kernel_sizes, list):
                    kernel_sizes = [kernel_sizes] * layers
                if not isinstance(ratio, list):
                    ratio = [ratio] * layers
                if not isinstance(recent_size, list):
                    recent_size = [recent_size] * layers
                for i in range(layers):
                    model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                    model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                    model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                    model.model.layers[i].self_attn.config.pooling = pooling
                    model.model.layers[i].self_attn.config.merge = args.merge
                    model.model.layers[i].self_attn.config.floor = args.floor
                    model.model.layers[i].self_attn.config.ratio = ratio[i]
                    model.model.layers[i].self_attn.config.recent_size = recent_size[i]
            
            # Prepare for generation
            tokenized_prompt = tokenizer(formatted_prompt, return_tensors="pt").to('cuda')
            input_ids = tokenized_prompt.input_ids
                        
            torch.cuda.synchronize()

            initial_nvidia_memory = get_gpu_memory()
            
            # --- Warmup Phase ---
            for _ in range(args.warmup_iters):
                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=tokenized_prompt.attention_mask, use_cache=True)
            
            print(f"Generating with {actual_length} input tokens...")
            
            # First, measure prefill time (time for first token generation)
            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            generation_start_event = torch.cuda.Event(enable_timing=True)
            generation_end_event = torch.cuda.Event(enable_timing=True)

            ######################
            # Benchmark Prefilling
            ######################
            past_key_values = None
            next_token = None
            prefill_start_event.record()

            # Prepare initial forward pass
            with torch.no_grad():
                if args.kv_quant is not None:
                    output = model.generate(
                        **tokenized_prompt,
                        max_new_tokens=1,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=tokenized_prompt['input_ids'].shape[-1]+1,
                        eos_token_id=[tokenizer.eos_token_id],
                        cache_implementation="quantized", 
                        cache_config=quant_cache_config,
                    )
                else:
                    output = model.generate(
                        **tokenized_prompt,
                        max_new_tokens=1,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=tokenized_prompt['input_ids'].shape[-1]+1,
                        eos_token_id=[tokenizer.eos_token_id]
                    )
            torch.cuda.synchronize()
            prefill_end_event.record()
            
            # Calculate prefill time
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0  # convert ms to seconds
            
            ######################
            # Benchmark Decoding
            ######################
            generation_start_event.record()
            with torch.no_grad():
                if args.kv_quant is not None:
                    output = model.generate(
                        **tokenized_prompt,
                        max_new_tokens=args.output_tokens,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=tokenized_prompt['input_ids'].shape[-1]+1,
                        eos_token_id=[tokenizer.eos_token_id],
                        cache_implementation="quantized", 
                        cache_config=quant_cache_config,
                    )
                else:
                    output = model.generate(
                        **tokenized_prompt,
                        max_new_tokens=args.output_tokens,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=tokenized_prompt['input_ids'].shape[-1]+1,
                        eos_token_id=[tokenizer.eos_token_id]
                    )
            torch.cuda.synchronize()
            generation_end_event.record()
            
            stop_flag.value = True
            monitor.join()
            
            # Calculate metrics for generation phase
            generation_time = generation_start_event.elapsed_time(generation_end_event) / 1000.0  # convert ms to seconds
            output_length = output[0].shape[0] - tokenized_prompt['input_ids'].shape[1]
            
            # Calculate tokens per second (only counting generated tokens)
            tokens_per_sec = output_length / (generation_time - prefill_time)
            
            # Store results
            length_results["model_memory_mb"].append(float(initial_nvidia_memory))
            length_results["total_memory_mb"].append(float(peak_memory_mb.value))
            length_results["prefill_time_sec"].append(float(prefill_time))
            length_results["decode_time_sec"].append(float(generation_time - prefill_time))
            length_results["tokens_per_sec"].append(float(tokens_per_sec))
            length_results["output_lengths"].append(int(output_length))
            
            print(f"Completed generation: {output_length} tokens in {generation_time:.2f} seconds")
            print(f"Metrics:")
            print(f"  - Total Memory: {(peak_memory_mb.value):.2f} MB")
            print(f"  - Model Memory: {initial_nvidia_memory:.2f} MB")
            print(f"  - Prefill Time: {prefill_time:.4f} sec")
            print(f"  - Decode Time: {generation_time - prefill_time:.4f} sec")
            print(f"  - Tokens/sec: {tokens_per_sec:.2f}")
            
            # Clear cache for next run but don't delete model
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.ipc_collect()
            outputs = None
        
        # # Calculate averages for this token length
        # length_results["avg_total_memory_mb"] = float(np.mean(length_results["total_memory_mb"]))
        # length_results["avg_prefill_time_sec"] = float(np.mean(length_results["prefill_time_sec"]))
        # length_results["avg_tokens_per_sec"] = float(np.mean(length_results["tokens_per_sec"]))
        
        # # Store results for this token length
        # benchmark_results["results"][str(token_length)] = length_results
        
        # print(f"\nAverage results for {token_length} tokens:")
        # print(f"  - Avg Total Memory: {length_results['avg_total_memory_mb']:.2f} MB")
        # print(f"  - Avg Prefill Time: {length_results['avg_prefill_time_sec']:.4f} sec")
        # print(f"  - Avg Tokens/sec: {length_results['avg_tokens_per_sec']:.2f}")
        
        # # Save interim results after each token length
        # results_file = os.path.join(output_dir, f"benchmark_results.json")
        # with open(results_file, "w") as f:
        #     json.dump(benchmark_results, f, indent=2)
            
    print(f"\nBenchmark complete! Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM memory usage and performance")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True, help="Path to the HuggingFace model")
    parser.add_argument("--model_max_length", type=int, default=32000, help="Maximum context length supported by the model")
    parser.add_argument("--method", type=str, default="FullKV", help="KV cache method (FullKV, PyramidKV, SnapKV, etc.)")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--kv_quant",type=str, default=None)
    parser.add_argument("--nbits", type=int, default=8, help="")
    
    # Benchmark parameters
    parser.add_argument("--custom_lengths", type=str, help="Comma-separated list of custom token lengths to test")
    parser.add_argument("--runs_per_length", type=int, default=3, help="Number of runs per token length for averaging")
    parser.add_argument("--output_tokens", type=int, default=512, help="Number of tokens to generate in each test")
    parser.add_argument("--save_dir", type=str, default="benchmark_results", help="Directory to save benchmark results")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Warmup Loops")
    
    # Optimized KV cache parameters
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="Maximum capacity for prompt tokens in optimized KV cache")
    parser.add_argument("--merge", type=str, default=None, help="KV merge method for PyramidKV")
    parser.add_argument("--floor", type=float, default=0.2, help="Floor parameter for AdaKV")
    parser.add_argument("--recent_size", type=int, default=32, help="Recent size parameter for optimized KV methods")
    parser.add_argument("--pruning_ratio", type=float, default=0.4, help="Pruning ratio for Key Cache")
    
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    main(args)