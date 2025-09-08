#!/bin/bash

PROJECT_DIR=path-to-root-project-dir

cd ${PROJECT_DIR}
export WORKING_DIR=${PROJECT_DIR}


bash ${PROJECT_DIR}/training_scripts/train_grpo-spec-sampling.sh \
    --dataset_name deepmath \
    --train_file_name train_sample_6144s_context_4k \
    --model_name Qwen3-1.7B-base \
    --max_response_length 4096 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --rollout_tp 2 \
    --rollout_name vllm \
    --save_freq 10 \
    --spec_decoding True \
    --bias 0.5 \
    --project_name your-project-name