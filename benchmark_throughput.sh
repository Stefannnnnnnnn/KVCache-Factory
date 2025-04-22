export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/workspace"

method=FullKV # Support PyramidKV, SnapKV, H2O, StreamingLLM, CAM, L2Norm, ThinK, FullKV
max_capacity_prompts=128 # 128,2048 in paper
attn_implementation=flash_attention_2 # Support "flash_attention_2", "sdpa", "eager".
source_path=~/outputs
model_path=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
merge_method=$7 # Support "pivot"(LOOK-M_PivotMerge).
nbits=4 # Quantization bit-width support 8,4,2. Need to set quant_method first.
save_dir=${source_path}"/results" # path to result save_dir

cache_dir="/workspace"

python3 benchmark_throughput.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --runs_per_length 1 \
    --warmup_iters 0 \
    --batch_size 4 \
    --kv_quant "HQQ" \
    --nbits ${nbits}
        # --merge ${merge_method} \