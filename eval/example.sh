

PROJECT_DIR=/home/work/compass_innovation/bsliu/Spec-RL
CKPT_DIR=/home/work/minzijun_rl_output/checkpoints/spec-rl_deepmath_train_sample_6144_context_4k_Qwen3-1.7B-base_max_response4096_batch1024_rollout8_vllm_n-2-w-6-4-start-6
BASE_MODEL_DIR=/home/work/minzijun_rl/models/Qwen3-1.7B-base

cd ${PROJECT_DIR}/eval/simplelr_math_eval

pip install pebble
cd latex2sympy
pip install -e .
cd ..
pip install word2number timeout_decorator jieba matplotlib sympy==1.13.1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash ${PROJECT_DIR}/eval/eval_math_nodes.sh \
    --run_name ${CKPT_DIR} \
    --init_model ${BASE_MODEL_DIR} \
    --template qwen-boxed  \
    --tp_size 1 \
    --add_step_0 false \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks amc23,math500,gsm8k,minerva_math,olympiadbench,mmlu_stem \
    --n_sampling 1 \
    --wandb_project demo-spec-rl