export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/workspace"

method=PyramidKV # Support PyramidKV, SnapKV, H2O, StreamingLLM, CAM, L2Norm, ThinK
max_capacity_prompts=128 # 128,2048 in paper
attn_implementation=flash_attention_2 # Support "flash_attention_2", "sdpa", "eager".
source_path=~/outputs
model_path=meta-llama/Meta-Llama-3-8B-Instruct
merge_method=$7 # Support "pivot"(LOOK-M_PivotMerge).
quant_method=kivi # Support kivi and kvquant, default None.
nbits=4 # Quantization bit-width support 8,4,2. Need to set quant_method first.
save_dir=${source_path}"results_long_bench" # path to result save_dir

cache_dir="/workspace"

python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --max_num_examples 1000 \
    --nbits ${nbits} \
    --quant_method ${quant_method} \
        # --merge ${merge_method} \