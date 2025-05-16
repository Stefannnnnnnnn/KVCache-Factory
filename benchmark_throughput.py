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

from awq import AutoAWQForCausalLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def generate_synthetic_prompt(target_length, tokenizer, variation_seed=None):
    """Generate a synthetic prompt with approximately the target number of tokens."""
    # Base text that can be repeated to achieve desired length
    base_text = "This is a benchmark test for language models. We are testing memory usage and generation speed with various input lengths. "
    
    # Add some variation if seed is provided
    if variation_seed is not None:
        random.seed(variation_seed)
        variations = [
            "The performance of large language models depends on many factors including hardware, implementation, and optimization techniques. ",
            "Benchmarking is essential to understand the trade-offs between speed, memory usage, and generation quality. ",
            "KV cache optimization is one way to improve inference efficiency while maintaining output quality. ",
            "Context length handling is becoming increasingly important as models are deployed in real-world applications. "
        ]
        base_text += random.choice(variations)
        
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
    
    # Hard trim to max length (to be safe)
    if len(encoded) > args.model_max_length - 5:
        encoded = encoded[:args.model_max_length - 5]
        synthetic_prompt = tokenizer.decode(encoded)
    
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

def safe_tokenize_batch(
    prompts: list,
    tokenizer,
    model_max_length: int,
    device: str = 'cuda',
    drop_empty: bool = True,
    verbose: bool = True,
):
    """
    Tokenizes a list of prompts with padding/truncation, and filters out all-padding inputs.

    Args:
        prompts (list): List of text prompts (str)
        tokenizer: HuggingFace tokenizer
        model_max_length (int): Max token length for the model
        device (str): Where to send the tensors ('cuda' or 'cpu')
        drop_empty (bool): Whether to remove all-padding inputs
        verbose (bool): Whether to print debug info

    Returns:
        tokenized_batch (dict): tokenized inputs suitable for model.generate
    """
    # Initial tokenization
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model_max_length,
        add_special_tokens=True,
    )

    # Identify non-empty inputs
    attention_mask = tokenized['attention_mask']
    valid_mask = attention_mask.sum(dim=1) > 0

    if drop_empty:
        # Filter only non-empty examples
        tokenized = {
            k: v[valid_mask] for k, v in tokenized.items()
        }

    if verbose:
        total = len(prompts)
        valid = valid_mask.sum().item() if drop_empty else total
        print(f"[safe_tokenize_batch] Total: {total}, Valid: {valid}, Dropped: {total - valid}")

    # Move to device
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    return tokenized

def main(args):
    # Dictionary to store benchmark results
    benchmark_results = {
        "model": args.model_path,
        "method": args.method,
        "batch_size": args.batch_size,
        "runs_per_length": args.runs_per_length,
        "output_tokens": args.output_tokens,
        "results": {}
    }
    
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
    output_dir = os.path.join(args.save_dir, f"{model_name}_{args.method}_batch{args.batch_size}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define token length scenarios to test
    token_lengths = [12000 / args.batch_size]
    if args.custom_lengths:
        token_lengths = [int(x) for x in args.custom_lengths.split(',')]
    
    # Filter lengths that exceed model's context window
    model_max_length = args.model_max_length
    valid_token_lengths = [l for l in token_lengths if l <= model_max_length]
    
    if len(valid_token_lengths) < len(token_lengths):
        skipped = set(token_lengths) - set(valid_token_lengths)
        print(f"Skipping token lengths {skipped} as they exceed model's max context length of {model_max_length}")
    
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

    print(f"Actual Attention class used in layer 0: {type(model.model.layers[0].self_attn)}")

    quant_cache_config={
        "nbits": args.nbits,
        "backend": "HQQ",
        "device": "cuda",
        "residual_length": args.output_tokens,
        "axis_key": 1,
        "q_group_size": 64
    }

    model.eval()
    
    # Test each token length
    for token_length in valid_token_lengths:
        print(f"\n{'='*40}")
        print(f"Testing with input length: {token_length} tokens, batch size: {args.batch_size}")
        print(f"{'='*40}")
        
        # Reset length_results for this token length
        length_results = {
            "prompts": [],
            "actual_input_lengths": [], # Renamed for clarity, stores individual input lengths
            "output_token_lengths": [], # Stores individual output lengths
            "model_memory_mb": [],
            "total_memory_mb": [],
            "prefill_time_sec": [],
            "decode_time_sec": [],
            "TTFT_sec": [], # Time To First Token
            "TPOT_sec_per_token": [], # Time Per Output Token
            "decode_throughput_tokens_per_sec": [], # Original tokens_per_sec, now clarified
            "total_throughput_tokens_per_sec": [], # Overall throughput
        }

        # Run multiple tests for this token length for more reliable results
        for run in range(args.runs_per_length):
            print(f"\nRun {run+1}/{args.runs_per_length} for {token_length} tokens, batch size: {args.batch_size}")
    
            # Start monitor process
            peak_memory_mb = Value('i', 0)
            stop_flag = Value('b', False)
            # Ensure memory_monitor is defined or mock it if not critical for this specific modification
            monitor = Process(target=memory_monitor, args=(peak_memory_mb, stop_flag))
            monitor.start()
    
            # Clear cache between runs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
            # Generate multiple synthetic prompts for batching
            batch_prompts = []
            batch_input_lengths = [] # To store actual lengths for all items in batch
            batch_formatted_prompts = []
    
            for batch_idx in range(args.batch_size):
                prompt, actual_length = generate_synthetic_prompt(token_length, tokenizer, variation_seed=run*args.batch_size+batch_idx)
                formatted_prompt = format_for_model(prompt, args.model_path)
    
                batch_prompts.append(prompt)
                batch_input_lengths.append(actual_length)
                batch_formatted_prompts.append(formatted_prompt)
    
            # Store the generated prompts for reference (just first one for simplicity)
            length_results["prompts"].append(batch_prompts[0][:200] + "..." if len(batch_prompts[0]) > 200 else batch_prompts[0])
            # Store the actual input length of the first item for reference / average later
            # For more accuracy, you might average batch_input_lengths if they can vary significantly
            current_input_length = batch_input_lengths[0]
            length_results["actual_input_lengths"].append(sum(batch_input_lengths))
    
            # Prepare batch for generation
            tokenized_prompts = safe_tokenize_batch(
                batch_formatted_prompts,
                tokenizer=tokenizer,
                model_max_length=args.model_max_length,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            input_ids = tokenized_prompts['input_ids']
    
            # Make sure KV cache is cleared
            if hasattr(model, 'past_key_values') and model.past_key_values is not None:
                model.past_key_values = None
    
            print("Preparing for next run...")
    
            ###########################################################################
            # The following part is for KV Cache Sparcification, currently using FullKV
            ###########################################################################
            if args.max_capacity_prompts != -1:
                max_capacity_prompts = args.max_capacity_prompts
            if args.method != "FullKV":
                if args.method.lower() in ["snapkv","pyramidkv","h2o","cam", "l2norm", "adakv", "headkv", "think"]:
                    window_sizes = 8
                elif args.method.lower() in ["streamingllm"]:
                    window_sizes = max_capacity_prompts - 4
                else: # Default window_sizes if not specified for a method
                    window_sizes = 8
    
    
                if args.method.lower() =='headkv':
                    # Ensure dummy head_path file exists for the dummy run or handle FileNotFoundError
                    try:
                        with open(args.head_path, 'r') as file:
                            head_list_data = json.loads(file.readline())
                        head_score_list = [np.mean(l[1]) for l in head_list_data.items()]
                        head_score_list = torch.tensor(head_score_list / sum(head_score_list))
                        total_attention = head_score_list.reshape(model.config.num_hidden_layers, model.config.num_attention_heads)
                        total_pool_capacity = (args.max_capacity_prompts // args.head_beta) * model.config.num_hidden_layers * model.config.num_attention_heads
                        min_num = (args.max_capacity_prompts - args.max_capacity_prompts // args.head_beta)
                        head_capacity = torch.round(total_attention * total_pool_capacity + min_num).int()
                        model.model.config.head_capacity = head_capacity
                    except FileNotFoundError:
                        print(f"Warning: Head path {args.head_path} not found. Skipping headkv specific setup.")
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {args.head_path}. Skipping headkv specific setup.")
    
    
                kernel_sizes = 7
                pooling = "maxpool"
                ratio = args.pruning_ratio
                recent_size = args.recent_size
    
                layers = len(model.model.layers)
                if not isinstance(window_sizes, list):
                    window_sizes = [window_sizes] * layers
                if not isinstance(max_capacity_prompts, list): # This variable might not be a list initially
                    max_capacity_prompts_list = [max_capacity_prompts] * layers
                else:
                    max_capacity_prompts_list = max_capacity_prompts
    
                if not isinstance(kernel_sizes, list):
                    kernel_sizes = [kernel_sizes] * layers
                if not isinstance(ratio, list):
                    ratio = [ratio] * layers
                if not isinstance(recent_size, list):
                    recent_size = [recent_size] * layers
    
                for i in range(layers):
                    model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                    model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts_list[i]
                    model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                    model.model.layers[i].self_attn.config.pooling = pooling
                    model.model.layers[i].self_attn.config.merge = args.merge
                    model.model.layers[i].self_attn.config.floor = args.floor
                    model.model.layers[i].self_attn.config.ratio = ratio[i]
                    model.model.layers[i].self_attn.config.recent_size = recent_size[i]
    
            # Get Initial(Model) Memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            initial_nvidia_memory = get_gpu_memory()
    
            # --- Warmup Phase ---
            if args.warmup_iters > 0 and torch.cuda.is_available(): # Added cuda check for warmup
                for _ in range(args.warmup_iters):
                    with torch.no_grad():
                        _ = model(**tokenized_prompts, use_cache=True) # Assuming model can take use_cache
    
            print(f"Generating with batch size {args.batch_size}, each with ~{current_input_length} input tokens...")
    
            prefill_start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else time.perf_counter
            prefill_end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else time.perf_counter
            generation_start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else time.perf_counter
            generation_end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else time.perf_counter
    
            ######################
            # Benchmark Prefilling
            ######################
            past_key_values = None
            next_token = None
            
            if torch.cuda.is_available():
                prefill_start_event.record()
            else:
                prefill_start_time_cpu = time.perf_counter()
    
            with torch.no_grad():
                if args.kv_quant is not None:
                    output = model.generate(
                        **tokenized_prompts, # Pass input_ids directly
                        output_attentions = False,
                        max_new_tokens=1,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=input_ids.shape[-1]+1, # Ensure at least one new token
                        eos_token_id=[tokenizer.eos_token_id],
                        cache_implementation="quantized",
                        cache_config=quant_cache_config,
                    )
                else:
                    output = model.generate(
                        **tokenized_prompts, # Pass input_ids directly
                        output_attentions = False,
                        max_new_tokens=1,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=input_ids.shape[-1]+1, # Ensure at least one new token
                        eos_token_id=[tokenizer.eos_token_id]
                    )
            
            if torch.cuda.is_available():
                prefill_end_event.record()
                torch.cuda.synchronize()
                prefill_time_sec = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
            else:
                prefill_time_sec = time.perf_counter() - prefill_start_time_cpu
    
    
            ######################
            # Benchmark Decoding
            ######################
            # For decoding, we want to measure the generation of args.output_tokens *after* the prefill.
            # The model.generate call for prefill already generates 1 token.
            # So for decode, we generate args.output_tokens - 1 more if prefill is counted,
            # or args.output_tokens if we are measuring a fresh generation call for the full sequence.
            # The current setup calls generate twice, once for prefill (1 token), once for decode (N tokens).
            # Let's stick to the original logic: first call is prefill, second is "full" generation including prefill.
            # The decode time will then be total_generation_time - prefill_time_for_first_token.
    
            if torch.cuda.is_available():
                generation_start_event.record()
            else:
                generation_start_time_cpu = time.perf_counter()
    
            with torch.no_grad():
                if args.kv_quant is not None:
                    print("Apply KV Quantization")
                    output = model.generate(
                        **tokenized_prompts, # Pass input_ids directly
                        output_attentions = False,
                        max_new_tokens=args.output_tokens,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=input_ids.shape[-1]+1, # Ensure at least one new token
                        eos_token_id=[tokenizer.eos_token_id],
                        cache_implementation="quantized",
                        cache_config=quant_cache_config,
                    )
                else:
                    output = model.generate(
                        **tokenized_prompts, # Pass input_ids directly
                        output_attentions = False,
                        max_new_tokens=args.output_tokens,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=input_ids.shape[-1]+1, # Ensure at least one new token
                        eos_token_id=[tokenizer.eos_token_id]
                    )
            
            if torch.cuda.is_available():
                generation_end_event.record()
                torch.cuda.synchronize()
                total_generation_time_sec = generation_start_event.elapsed_time(generation_end_event) / 1000.0
            else:
                total_generation_time_sec = time.perf_counter() - generation_start_time_cpu
    
    
            stop_flag.value = True
            monitor.join() # Wait for memory monitor to finish
    
            # Calculate output lengths - might be different for each item in batch
            output_lengths = []
            for i, output_seq in enumerate(output):
                # output_length is number of *newly* generated tokens
                output_length = output_seq.shape[0] - input_ids.shape[1]
                output_lengths.append(output_length)
    
            avg_output_length = sum(output_lengths) / len(output_lengths) if output_lengths else 0
            total_generated_tokens = sum(output_lengths)
    
            # Refined time calculations
            decode_time_sec = total_generation_time_sec - prefill_time_sec
            if decode_time_sec < 0: # Prefill might be a significant portion of total if output_tokens is small
                decode_time_sec = 0 # Avoid negative if total_time is very close to prefill_time due to timing precision
    
            # Calculate metrics
            TTFT_sec = prefill_time_sec
            TPOT_sec_per_token = decode_time_sec / avg_output_length if avg_output_length > 0 else 0
            decode_throughput_tokens_per_sec = total_generated_tokens / decode_time_sec if decode_time_sec > 0 else 0
            total_throughput_tokens_per_sec = (total_generated_tokens + sum(batch_input_lengths)) / total_generation_time_sec if total_generation_time_sec > 0 else 0
    
    
            # Store results
            length_results["model_memory_mb"].append(float(initial_nvidia_memory))
            length_results["total_memory_mb"].append(float(peak_memory_mb.value))
            length_results["prefill_time_sec"].append(float(prefill_time_sec))
            length_results["decode_time_sec"].append(float(decode_time_sec))
            length_results["output_token_lengths"].append(int(avg_output_length)) # Storing avg output length for this run
    
            length_results["TTFT_sec"].append(float(TTFT_sec))
            length_results["TPOT_sec_per_token"].append(float(TPOT_sec_per_token))
            length_results["decode_throughput_tokens_per_sec"].append(float(decode_throughput_tokens_per_sec))
            length_results["total_throughput_tokens_per_sec"].append(float(total_throughput_tokens_per_sec))
    
    
            print(f"Completed batch generation: {total_generated_tokens} total output tokens ({avg_output_length:.1f} avg per batch item)")
            print(f"Input token length: {sum(batch_input_lengths)}")
            print(f"Metrics for this run:")
            print(f"  - Total Memory: {peak_memory_mb.value:.2f} MB")
            print(f"  - Model Memory: {initial_nvidia_memory:.2f} MB")
            print(f"  - Prefill Time (TTFT): {prefill_time_sec:.4f} sec")
            print(f"  - Decode Time: {decode_time_sec:.4f} sec")
            print(f"  - TPOT (Time Per Output Token): {TPOT_sec_per_token:.6f} sec/token")
            print(f"  - Decode Throughput: {decode_throughput_tokens_per_sec:.2f} tokens/sec")
            print(f"  - Total Throughput (incl. prefill): {total_throughput_tokens_per_sec:.2f} tokens/sec")
    
            # Clear cache for next run but don't delete model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.ipc_collect() # This might not be available on all PyTorch versions or needed
            outputs = None
            output = None # Clear output tensor
    
        # Calculate averages for this token length
        length_results["avg_input_token_length"] = float(np.mean(length_results["actual_input_lengths"]))
        length_results["avg_output_token_length"] = float(np.mean(length_results["output_token_lengths"]))
        length_results["avg_total_memory_mb"] = float(np.mean(length_results["total_memory_mb"]))
        length_results["avg_prefill_time_sec"] = float(np.mean(length_results["prefill_time_sec"])) # This is also avg TTFT
        length_results["avg_decode_time_sec"] = float(np.mean(length_results["decode_time_sec"]))
        length_results["avg_TTFT_sec"] = float(np.mean(length_results["TTFT_sec"]))
        length_results["avg_TPOT_sec_per_token"] = float(np.mean(length_results["TPOT_sec_per_token"]))
        length_results["avg_decode_throughput_tokens_per_sec"] = float(np.mean(length_results["decode_throughput_tokens_per_sec"]))
        length_results["avg_total_throughput_tokens_per_sec"] = float(np.mean(length_results["total_throughput_tokens_per_sec"]))
    
        # Store results for this token length
        benchmark_results["results"][str(token_length)] = length_results
    
        print(f"\nAverage results for input token length ~{length_results['avg_input_token_length']:.0f} (target {token_length}) with batch size {args.batch_size}:")
        print(f"  - Avg Output Token Length: {length_results['avg_output_token_length']:.1f}")
        print(f"  - Avg Total Memory: {length_results['avg_total_memory_mb']:.2f} MB")
        print(f"  - Avg Prefill Time (TTFT): {length_results['avg_prefill_time_sec']:.4f} sec")
        print(f"  - Avg Decode Time: {length_results['avg_decode_time_sec']:.4f} sec")
        print(f"  - Avg TPOT (Time Per Output Token): {length_results['avg_TPOT_sec_per_token']:.6f} sec/token")
        print(f"  - Avg Decode Throughput: {length_results['avg_decode_throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"  - Avg Total Throughput (incl. prefill): {length_results['avg_total_throughput_tokens_per_sec']:.2f} tokens/sec")

        # Save interim results after each token length
        results_file = os.path.join(output_dir, f"benchmark_results_{args.method}.json") # Added method to filename
        with open(results_file, "w") as f:
            json.dump(benchmark_results, f, indent=2)
    
    print(f"\nBenchmark complete! Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM memory usage and performance")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True, help="Path to the HuggingFace model")
    parser.add_argument("--model_max_length", type=int, default=32000, help="Maximum context length supported by the model")
    parser.add_argument("--method", type=str, default="FullKV", help="KV cache method (FullKV, PyramidKV, SnapKV, etc.)")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--kv_quant",type=str, default=None)
    parser.add_argument("--nbits", type=int, default=8, help="")
    
    # Benchmark parameters
    parser.add_argument("--custom_lengths", type=str, help="Comma-separated list of custom token lengths to test")
    parser.add_argument("--runs_per_length", type=int, default=3, help="Number of runs per token length for averaging")
    parser.add_argument("--output_tokens", type=int, default=1024, help="Number of tokens to generate in each test")
    parser.add_argument("--save_dir", type=str, default="benchmark_results", help="Directory to save benchmark results")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Warmup Loops")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    
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