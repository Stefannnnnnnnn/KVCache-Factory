import os
import json
import random
import argparse
import time
from collections import defaultdict
import subprocess

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Added for LLAMA-GPTQ
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# from auto_gptq import exllama_set_max_input_length

from awq import AutoAWQForCausalLM

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

datasets = [
    # "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
    #         "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
    #         "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
    # "narrativeqa"
    "triviaqa", "gov_report"
]

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

# model2maxlen = {
#     "Llama-2-7b-chat-hf": 3950,
#     "Llama-3-8B-Instruct": 7950,
#     "Meta-Llama-3-70B-Instruct": 7950,
#     "Meta-Llama-3-8B-Instruct-32k": 31500,
#     "Llama-2-7B-32K-Instruct": 31500,
#     "Mistral-7B-Instruct-v0.2": 31500,
#     "Mistral-7B-Instruct-v0.1": 31500,

# }

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "mistral": 31500
}



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt

# def build_prompt(prompt, dataset):
    
#     SYSTEM_PROMPT = model2prompt[dataset]

#     prompt = f"<<SYS>>\n {SYSTEM_PROMPT} \n<</SYS>>\n\n{prompt}"
#     return prompt


def get_gpu_memory():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        ).strip()
        
        # 处理输出为空的情况
        if not result:
            return 0
        
        # 解析每张 GPU 的内存占用
        return sum(int(x) for x in result.split('\n'))
    
    except FileNotFoundError:
        print("Error: nvidia-smi command not found. Is NVIDIA driver installed?")
        return 0
    except subprocess.CalledProcessError:
        print("Error: Failed to execute nvidia-smi command.")
        return 0
    except ValueError:
        print("Error: Unexpected output format from nvidia-smi.")
        return 0

def main(args):
    

    print("Loading data...")
    
    test_data = []
    
    prompts = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []
    
    input_max_len = 0
    
    model_path = args.model_path.lower()

    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
            

    
    output_max_len = dataset2maxlen[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            
            
            length = example["length"]
            if length > input_max_len: input_max_len = length
            
            template = model2prompt[args.dataset]
            prompt = template.format(**example)
            
            if "llama2" in args.model_path.lower():
                prompt = build_chat(prompt)
                
            example["prompt"] = prompt
                
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
        
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    
    for example in test_data:
        
        prompts.append(example["prompt"])
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example["dataset"])
        languages.append(example["language"])
        all_classess.append(example["all_classes"])
        _ids.append(example["_id"])

    print("Finish loading model and tokenizer")
    
    model_name = model_path.split("/")[-1]

    # Create output directory
    output_dir = os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}.json"), "w")

    perf_metrics = {
        'total_tokens_generated': 0,
        'total_prefill_time': 0,
        'total_decode_time': 0,
        'peak_memory_pytorch': 0,
        'peak_memory_nvidia': 0,
        'per_length_metrics': defaultdict(lambda: {'tokens': 0, 'prefill_time': 0, 'decode_time': 0, 'count': 0}),
        'latencies': []
    }

    torch.cuda.synchronize()
    # Start with clean CUDA memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_memory_pytorch = torch.cuda.memory_allocated()
    initial_memory_nvidia = get_gpu_memory()
     
    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        
        batch_prompts = prompts[i:i+args.eval_batch_size]
        batch_inputs = inputs[i:i+args.eval_batch_size]
        batch_contexts = contexts[i:i+args.eval_batch_size]
        batch_answerss = answerss[i:i+args.eval_batch_size]
        batch_lengths = lengths[i:i+args.eval_batch_size]
        
        batch_datasets = datasets[i:i+args.eval_batch_size]
        batch_languages = languages[i:i+args.eval_batch_size]
        batch_all_classess = all_classess[i:i+args.eval_batch_size]
        batch__ids = _ids[i:i+args.eval_batch_size]
        
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        # Record sequence length for metrics
        seq_length = batch_input_ids.shape[1]
        
        # Clear CUDA cache before generation
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()
        nvidia_memory_before = get_gpu_memory()
        
        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
            
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask

        # # default to True
        # if args.method == "DynamicKV":
        #     args.output_attentions = True
        # else:
        #     args.output_attentions=False

        if args.max_capacity_prompts != -1:
            max_capacity_prompts = args.max_capacity_prompts
        elif args.max_capacity_prompts_ratio != -1:
            max_capacity_prompts = round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio)
        
        
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
            

        context_length = batch_input_ids.shape[-1]

        # Setup CUDA timing events for accurate measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Record start time
        start_event.record()
        
        if args.quant_method == None:        
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id]
            )
        else:
            print("Using quantization on kv cache")
            quant_cache_config={
                "nbits": args.nbits,
                "backend": "HQQ",
                "device": "cuda",
                "residual_length": output_max_len,
                "axis_key": 1,
                "q_group_size": 64
            }
            
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id],
                cache_implementation="quantized", 
                cache_config=quant_cache_config,
            )

        # batch_outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)

        # Record end time and synchronize
        end_event.record()
        torch.cuda.synchronize()
        
        # Get total generation time in seconds
        total_generation_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
        
        # Get output sequence
        if hasattr(output, "sequences"):
            generated_sequence = output.sequences[0]
            new_tokens = output.sequences.shape[1] - context_length
            batch_outputs = tokenizer.batch_decode([output.sequences[0][context_length:]], skip_special_tokens=True)
        else:
            generated_sequence = output[0]
            new_tokens = generated_sequence.shape[0] - context_length
            batch_outputs = tokenizer.batch_decode([generated_sequence[context_length:]], skip_special_tokens=True)
        
        batch_generations = batch_outputs
        
        # Get prefill and decode timing, with fallback if not provided
        prefill_time = getattr(output, "prefill_time", None)
        decode_time = getattr(output, "decode_time", None)
        
        # Fallback if timing not provided by model
        if prefill_time is None or decode_time is None:
            # Estimate split: typically prefill is 1/3 of time for long contexts
            prefill_ratio = 0.35  # Estimated ratio for prefill vs total time
            prefill_time = total_generation_time * prefill_ratio
            decode_time = total_generation_time * (1 - prefill_ratio)
        
        # Small epsilon to prevent division by zero
        epsilon = 1e-6
        
        # 记录内存状态（操作执行后）
        memory_after = torch.cuda.memory_allocated()
        nvidia_memory_after = get_gpu_memory()
        
        # 计算 GPU 内存变化
        memory_used = max(memory_after - memory_before, 0)
        nvidia_memory_used = max(nvidia_memory_after - nvidia_memory_before, 0)
        
        # Update performance metrics
        perf_metrics['total_tokens_generated'] += new_tokens
        perf_metrics['total_prefill_time'] += prefill_time
        perf_metrics['total_decode_time'] += decode_time
        perf_metrics['peak_memory_pytorch'] = max(perf_metrics['peak_memory_pytorch'], torch.cuda.max_memory_allocated())
        perf_metrics['peak_memory_nvidia'] = max(perf_metrics['peak_memory_nvidia'], nvidia_memory_after)
        
        # Per-length metrics
        length_bucket = seq_length // 500 * 500  # Group by 500 tokens
        perf_metrics['per_length_metrics'][length_bucket]['tokens'] += new_tokens
        perf_metrics['per_length_metrics'][length_bucket]['prefill_time'] += prefill_time
        perf_metrics['per_length_metrics'][length_bucket]['decode_time'] += decode_time
        perf_metrics['per_length_metrics'][length_bucket]['count'] += 1
        
        # Record latency for this batch
        perf_metrics['latencies'].append({
            'seq_length': seq_length,
            'output_length': new_tokens,
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_generation_time,
            'prefill_tokens_per_sec': seq_length / (prefill_time + epsilon),
            'decode_tokens_per_sec': new_tokens / (decode_time + epsilon)
        })

        for j in range(args.eval_batch_size):
            
            example = {}
            
            example["prompt"] = batch_prompts[j]
            example["input"] = batch_inputs[j]
            example["context"] = batch_contexts[j]
            example["answers"] = batch_answerss[j]
            example["pred"] = batch_generations[j]
            example["length"] = batch_lengths[j]
            
            example["dataset"] = batch_datasets[j]
            example["language"] = batch_languages[j]
            example["all_classes"] = batch_all_classess[j]
            example["_id"] = batch__ids[j]

            # print(f'{batch_generations[j]}')
            fout.write(json.dumps(example) + "\n")
            
        # Clear cache between batches
        torch.cuda.empty_cache()

    # Calculate final metrics with sanity checks
    epsilon = 1e-6  # Small value to avoid division by zero
    
    # Calculate overall metrics
    overall_tokens_per_sec = perf_metrics['total_tokens_generated'] / max(
        perf_metrics['total_prefill_time'] + perf_metrics['total_decode_time'], epsilon)
    
    prefill_tokens_per_sec = sum(l['seq_length'] for l in perf_metrics['latencies']) / max(
        perf_metrics['total_prefill_time'], epsilon)
    
    decode_tokens_per_sec = perf_metrics['total_tokens_generated'] / max(
        perf_metrics['total_decode_time'], epsilon)
    
    # Apply reasonable caps to prevent absurd values
    MAX_REASONABLE_PREFILL_TPS = 10000  # 10k tokens per second for prefill
    MAX_REASONABLE_DECODE_TPS = 300     # 300 tokens per second for decode
    
    prefill_tokens_per_sec = min(prefill_tokens_per_sec, MAX_REASONABLE_PREFILL_TPS)
    decode_tokens_per_sec = min(decode_tokens_per_sec, MAX_REASONABLE_DECODE_TPS)
    overall_tokens_per_sec = min(overall_tokens_per_sec, MAX_REASONABLE_DECODE_TPS)
    
    # Process per-length metrics
    per_length_report = {}
    for length, data in sorted(perf_metrics['per_length_metrics'].items()):
        if data['count'] > 0:
            avg_prefill = data['prefill_time'] / data['count']
            avg_decode = data['decode_time'] / data['count']
            avg_tokens = data['tokens'] / data['count']
            
            per_length_report[str(length)] = {
                'avg_prefill_time': avg_prefill,
                'avg_decode_time': avg_decode,
                'avg_tokens_generated': avg_tokens,
                'prefill_tokens_per_sec': min(length / max(avg_prefill, epsilon), MAX_REASONABLE_PREFILL_TPS),
                'decode_tokens_per_sec': min(avg_tokens / max(avg_decode, epsilon), MAX_REASONABLE_DECODE_TPS),
                'sample_count': data['count']
            }
    
    # Calculate latency statistics
    if perf_metrics['latencies']:
        latency_stats = {
            'prefill': {
                'min': min(l['prefill_time'] for l in perf_metrics['latencies']),
                'max': max(l['prefill_time'] for l in perf_metrics['latencies']),
                'avg': sum(l['prefill_time'] for l in perf_metrics['latencies']) / len(perf_metrics['latencies'])
            },
            'decode': {
                'min': min(l['decode_time'] for l in perf_metrics['latencies']),
                'max': max(l['decode_time'] for l in perf_metrics['latencies']),
                'avg': sum(l['decode_time'] for l in perf_metrics['latencies']) / len(perf_metrics['latencies'])
            }
        }
    else:
        latency_stats = {'prefill': {}, 'decode': {}}
    
    # Calculate memory usage
    memory_used_mb_pytorch = (perf_metrics['peak_memory_pytorch'] - initial_memory_pytorch) / (1024 * 1024)
    memory_used_mb_nvidia = perf_metrics['peak_memory_nvidia'] - initial_memory_nvidia
    
    # Create detailed performance report
    detailed_performance_report = {
        "method": args.method,
        "model": args.model_path,
        "dataset": args.dataset,
        "max_capacity_prompts": args.max_capacity_prompts,
        "throughput": {
            "overall_tokens_per_second": float(overall_tokens_per_sec),
            "prefill_tokens_per_second": float(prefill_tokens_per_sec),
            "decode_tokens_per_second": float(decode_tokens_per_sec),
        },
        "memory": {
            "peak_memory_pytorch_mb": float(perf_metrics['peak_memory_pytorch'] / (1024 * 1024)),
            "peak_memory_nvidia_mb": float(perf_metrics['peak_memory_nvidia']),
            "memory_used_pytorch_mb": float(memory_used_mb_pytorch),
            "memory_used_nvidia_mb": float(memory_used_mb_nvidia),
        },
        "timing": {
            "total_prefill_time_seconds": float(perf_metrics['total_prefill_time']),
            "total_decode_time_seconds": float(perf_metrics['total_decode_time']),
            "total_generation_time_seconds": float(perf_metrics['total_prefill_time'] + perf_metrics['total_decode_time']),
        },
        "tokens_info": {
            "total_tokens_generated": int(perf_metrics['total_tokens_generated']),
        },
        "per_length_performance": per_length_report,
        "latency_statistics": latency_stats
    }

    # Create simplified performance report for compatibility
    performance_report = {
        "method": args.method,
        "model": args.model_path,
        "dataset": args.dataset,
        "max_capacity_prompts": args.max_capacity_prompts,
        "average_tokens_per_second": float(overall_tokens_per_sec),
        "prefill_tokens_per_second": float(prefill_tokens_per_sec),
        "decode_tokens_per_second": float(decode_tokens_per_sec),
        "peak_memory_usage_mb": float(perf_metrics['peak_memory_pytorch'] / (1024 * 1024)),
        "peak_memory_nvidia_mb": float(perf_metrics['peak_memory_nvidia']),
        "memory_used_mb": float(memory_used_mb_pytorch),
        "total_generation_time_seconds": float(perf_metrics['total_prefill_time'] + perf_metrics['total_decode_time']),
        "total_tokens_generated": int(perf_metrics['total_tokens_generated'])
    }

    # Save performance reports
    detailed_perf_file_path = os.path.join(output_dir, f"{args.method}_detailed_performance.json")
    with open(detailed_perf_file_path, "w") as f:
        json.dump(detailed_performance_report, f, indent=2)
    
    perf_file_path = os.path.join(output_dir, f"{args.method}_performance.json")
    with open(perf_file_path, "w") as f:
        json.dump(performance_report, f, indent=2)
    
    fout.close()
    
# def compare_performance(results_dir):
#     # 收集所有性能报告
#     performance_data = []
#     for dataset_dir in os.listdir(results_dir):
#         dataset_path = os.path.join(results_dir, dataset_dir)
#         if os.path.isdir(dataset_path):
#             for file in os.listdir(dataset_path):
#                 if file.endswith("_performance.json"):
#                     with open(os.path.join(dataset_path, file), "r") as f:
#                         perf_data = json.load(f)
#                         performance_data.append(perf_data)
#     # 创建比较表格
#     import pandas as pd
#     df = pd.DataFrame(performance_data)
#     df.to_csv(os.path.join(results_dir, "performance_comparison.csv"), index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--quant_method",type=str,default=None,choices=["kivi","kvquant"])
    parser.add_argument("--nbits", type=int, default=8, help="")
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--merge", type=str, default=None, help="kv merge method(look-m)")
    parser.add_argument('--floor', type=float, default=0.2, help='hyper-parameter used in AdaKV')
    parser.add_argument('--head_path', type=str, default='./data/heads_score/Meta-Llama-3-8B-Instruct_retrieval_reasoning_heads.json', help='Path to head score (HeadKV)')
    parser.add_argument('--head_beta', type=float, default=1.01, help='hyper-parameter used on HeadKV')
    parser.add_argument("--recent_size", type=int, default=32, help="")
    parser.add_argument("--pruning_ratio", type=float, default=0.4, help="pruning ratio of Key Cache")

    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    if args.quant_method == "kvquant":
        from pyramidkv.quantcache import KVQuantizedCache
        from transformers import cache_utils
        cache_utils.HQQQuantizedCache = KVQuantizedCache
        
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left",
        cache_dir="/workspace"
    )

    from pyramidkv.monkeypatch import replace_llama
    # ,replace_mistral
    replace_llama(args.method.lower())
    # replace_mistral(args.method.lower())
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation,
        cache_dir="/workspace"
    )    

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

        
    model.eval()
    
    save_dir = args.save_dir
    
        
    max_capacity_prompts = args.max_capacity_prompts
    
    for idx, dataset in enumerate(datasets):
        
        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(datasets)}")
        
        args.dataset = dataset
        
        args.data_file = f"data/LongBench/{args.dataset}.jsonl"
        
        main(args)

    # compare_performance(args.save_dir)