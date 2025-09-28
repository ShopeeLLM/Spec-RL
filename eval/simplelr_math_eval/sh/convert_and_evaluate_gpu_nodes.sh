#!/bin/bash
set -x

# 参数检查
if [ "$#" -gt 13 ]; then
    echo "Usage: $0 <eval_script_path> <base_checkpoint_path> <init_model_path> <template> [benchmarks] [temperature] [max_tokens] [top_p] [tp_size] [ckpt_list_file] [output_dir] [overwrite] [n_sampling]"
    exit 1
fi

# 获取参数
eval_script_path=$1
base_checkpoint_path=$2
init_model_path=$3
template=$4
benchmarks=$5
temperature=$6
max_tokens=$7
top_p=$8
tp_size=${9:-1} 
ckpt_list_file=${10:-""} 
output_dir=${11:-"eval_results"}
overwrite=${12:-false}
n_sampling=${13:-1}
actor_dir="actor"


if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    total_phys=$(nvidia-smi -L 2>/dev/null | wc -l)
    if (( total_phys == 0 )); then
        echo "[FATAL] 没检测到 GPU, 且未设置 CUDA_VISIBLE_DEVICES" >&2
        exit 1
    fi
    # 默认全卡可见：0,1,2,...
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((total_phys-1)))
fi
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 拆成数组
IFS=',' read -r -a GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARR[@]}

if (( NUM_GPUS < tp_size )); then
    echo "[FATAL] tp_size ($tp_size) > NUM_GPUS ($NUM_GPUS)" >&2
    exit 1
fi

NUM_GPU_GROUPS=$((NUM_GPUS / tp_size))
echo "[INFO] NUM_GPUS=$NUM_GPUS  =>  NUM_GPU_GROUPS=$NUM_GPU_GROUPS (tp_size=$tp_size)"


# 函数：复制 tokenizer 文件
copy_tokenizer_files() {
    local ckpt_path=$1
    local init_model_path=$2
    local files_to_copy=(
        "added_tokens.json"
        "config.json"
        "generation_config.json"
        "special_tokens_map.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
    )
    if [ -f "$init_model_path/merges.txt" ]; then
        files_to_copy+=("merges.txt")
    fi
    # 创建目标路径，确保它存在
    if [ ! -d "$ckpt_path" ]; then
        mkdir -p "$ckpt_path"
        echo "Created checkpoint directory: $ckpt_path" >&2
    else
        echo "Checkpoint directory already exists: $ckpt_path" >&2
    fi

    # 复制每个文件
    for filename in "${files_to_copy[@]}"; do
        src="$init_model_path/$filename"
        dst="$ckpt_path/$filename"
        if [ -e "$src" ]; then
            cp "$src" "$dst"
            echo "Copied $src to $dst"
        else
            echo "Warning: $src does not exist."
        fi
    done
}

# 函数：获取所有需要评估的检查点，并过滤掉已评估的
get_checkpoints_to_evaluate() {
    local base_path="$1"
    
    if [ -n "$ckpt_list_file" ] && [ -f "$ckpt_list_file" ]; then
        # Read checkpoints from the provided file
        cat "$ckpt_list_file"
    else
        # Original logic for getting all checkpoints
        local checkpoints=()
        for ckpt_dir in "$base_path"/global_step_*; do
            if [ -d "$ckpt_dir" ]; then
                step_tag=$(basename "$ckpt_dir")
                checkpoints+=("$step_tag")
            fi
        done
        
        if [ ${#checkpoints[@]} -eq 0 ]; then
            echo ""
        else
            printf "%s\n" "${checkpoints[@]}"
        fi
    fi
}

# 函数：在指定GPU上处理单个检查点
process_checkpoint() {
    local step_tag=$1   # e.g. global_step_40
    local group_id=$2   # 第几组显卡

    local total_visible=${#GPU_ARR[@]}
    local start_idx=$((group_id * tp_size))

    if (( start_idx + tp_size > total_visible )); then
        echo "[ERROR] group $group_id 需要 $tp_size 张卡，但可见卡只有 $total_visible 张！" >&2
        return 1
    fi

    # 组装 GPU 列表
    local gpu_ids=""
    for ((i=0; i<tp_size; i++)); do
        gpu_ids+="${gpu_ids:+,}${GPU_ARR[$((start_idx + i))]}"
    done

    local ckpt_path="$base_checkpoint_path/$step_tag/actor/huggingface"
    local output_path_new="$base_checkpoint_path/$output_dir/$step_tag"

    mkdir -p "$output_path_new"

    echo "[RUN ] $step_tag  ->  GPU[$gpu_ids]  ->  $output_path_new" >&2

    CUDA_VISIBLE_DEVICES="$gpu_ids" \
    bash "$eval_script_path" \
         "$template" \
         "$ckpt_path" \
         "$output_path_new" \
         "$temperature" "$max_tokens" "$top_p" \
         "$benchmarks" "$overwrite" "$n_sampling"
}

# 记录当前工作目录
original_dir=$(pwd)

# 主脚本部分修改
# 获取需要评估的检查点
readarray -t checkpoints_to_evaluate < <(get_checkpoints_to_evaluate "$base_checkpoint_path")

if [ ${#checkpoints_to_evaluate[@]} -eq 0 ]; then
    echo "No new checkpoints to evaluate." >&2
    exit 0
fi

# 检查GPU数量是否满足tp_size要求
if [ $((NUM_GPUS % tp_size)) -ne 0 ]; then
    echo "Error: Number of available GPUs ($NUM_GPUS) is not divisible by tp_size ($tp_size)" >&2
    exit 1
fi

tmp_fifo=$(mktemp -u)
mkfifo "$tmp_fifo"
exec 9<>"$tmp_fifo"
rm "$tmp_fifo"             # 文件名删掉，FD 仍可用

# 把 0 … NUM_GPU_GROUPS-1 放进令牌池
for ((g=0; g<NUM_GPU_GROUPS; g++)); do
  echo "$g" >&9
done

for step_tag in "${checkpoints_to_evaluate[@]}"; do
  # 阻塞式读一个可用 group_id（令牌）
  read -u 9 group_id
  {
    process_checkpoint "$step_tag" "$group_id"
    # 归还令牌
    echo "$group_id" >&9
  } &
done

wait           # 等全部后台任务
exec 9>&-      # 关闭 fifo
echo "All conversions and evaluations completed."
