

PROJECT_DIR=path-to-your-base-project-dir
CKPT_DIR=path-to-your-ckpt-to-be-evaluated
BASE_MODEL_DIR=path-to-your-base-model

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