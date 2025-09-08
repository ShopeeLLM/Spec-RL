set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=your-wandb-api-key

export RUN_NAME=your-custom-run-name
export DATA_PATH=path-to-data-dir
export LOG_PATH=path-to-log-dir
mkdir -p $LOG_PATH


# Default values
PROJECT_NAME=verl-0.5-train
TRAIN_FILE_NAME=train
TRAIN_BATCH_SIZE=1024
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=4096
LEARNING_RATE=5e-7
PPO_MINI_BATCH_SIZE=256
PPO_MICRO_BATCH_SIZE=2
KL_LOSS_COEF=0.0001
KL_COEF=0.001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=16
ROLLOUT_N=8
ROLLOUT_NAME=vllm
ROLLOUT_GPU_MEMORY_UTIL=0.7
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
TOTAL_EPOCHS=15
TOTAL_STEPS=110
DATASET_NAME=simplelr_math_35
MODEL_NAME=Qwen2.5-Math-7B
SAVE_FREQ=10
TEST_FREQ=10
NUM_GPU=8
REWARD_FN_PATH=path-to-reward-fn
MODEL_PATH=path-to-default-base-model-dir
CHECKPOINT_PATH=path-to-default-save-model-path
mkdir -p $CHECKPOINT_PATH


generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local model_provided=false
  local suffix_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --val_batch_size) suffix+="_valbatch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_max_prompt$2"; shift 2 ;;
      --max_response_length) suffix+="_max_response$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --kl_loss_coef) suffix+="_klcoef$2"; shift 2 ;;
      --entropy_coeffient) suffix+="_entcoef$2"; shift 2 ;;
      --clip_ratio) suffix+="_clipratio$2"; shift 2 ;;
      --kl_loss_type) suffix+="_kltype$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --log_prob_micro_batch_size) suffix+="_logprobbatch$2"; shift 2 ;;
      --rollout_n) suffix+="_rollout$2"; shift 2 ;;
      --rollout_name) suffix+="_$2"; shift 2 ;;
      --kl_coef) suffix+="_klcontrol$2"; shift 2 ;;
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --model_name) model_name_sanitized=$(echo "$2" | tr '/' '_'); suffix+="_$model_name_sanitized"; model_provided=true; shift 2 ;;
      --remove_clip) suffix+="_remove_clip$2"; shift 2 ;;
      --train_file_name) suffix+="_$2"; shift 2 ;;
      --suffix) input_suffix="$2"; suffix_provided=true; shift 2 ;;
      *) shift ;;
    esac
  done

  if [ "$dataset_provided" = false ]; then
    suffix+="_$DATASET_NAME"
  fi

  if [ "$model_provided" = false ]; then
    model_name_sanitized=$(echo "$MODEL_NAME" | tr '/' '_')
    suffix+="_$model_name_sanitized"
  fi

  if [ "$suffix_provided" = true ]; then
    suffix+="_$input_suffix"
  fi
  
  echo "$suffix"
}

echo "Arguments received: $@"


SUFFIX=$(generate_suffix "$@")
RUN_NAME="$RUN_NAME$SUFFIX"
LOG_FILE_PATH="$LOG_PATH/$RUN_NAME.log"


# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --ppo_micro_batch_size) PPO_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --entropy_coeffient) ENTROPY_COEFFIENT="$2"; shift 2 ;;
    --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout_name) ROLLOUT_NAME="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --rollout_tp) ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --kl_coef) KL_COEF="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --total_steps) TOTAL_STEPS="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --test_freq) TEST_FREQ="$2"; shift 2 ;;
    --train_file_name) TRAIN_FILE_NAME="$2"; shift 2 ;;
    --num_gpu) NUM_GPU="$2"; shift 2 ;;
    --project_name) PROJECT_NAME="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done


echo "Training with the following parameters:"
echo "Train File Name: $TRAIN_FILE_NAME.parquet"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Max Prompt Length: $MAX_PROMPT_LENGTH" 
echo "Max Response Length: $MAX_RESPONSE_LENGTH" 
echo "Learning Rate: $LEARNING_RATE" 
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE" 
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE" 
echo "KL Loss Coefficient: $KL_LOSS_COEF" 
echo "KL Loss Type: $KL_LOSS_TYPE" 
echo "Rollout N: $ROLLOUT_N" 
echo "KL Coefficient: $KL_COEF" 
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Total Training Steps: $TOTAL_STEPS"
echo "Dataset Name: $DATASET_NAME"
echo "Model Name: $MODEL_NAME"
echo "Log File Path: $LOG_FILE_PATH"
echo "Rollout Name: $ROLLOUT_NAME"
echo "Num GPU: $NUM_GPU"
echo "Checkpoint Path: $CHECKPOINT_PATH"

mkdir -p $CHECKPOINT_PATH/$RUN_NAME

export RAY_memory_usage_threshold=0.99

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/$DATASET_NAME/$TRAIN_FILE_NAME.parquet \
    data.val_files=$DATA_PATH/$DATASET_NAME/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH/$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    custom_reward_function.path=$REWARD_FN_PATH \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.rollout_data_dir=rollouts/$RUN_NAME \
    trainer.n_gpus_per_node=$NUM_GPU \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_local_dir=$CHECKPOINT_PATH/$RUN_NAME \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee -a $LOG_FILE_PATH
