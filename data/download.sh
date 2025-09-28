#!/bin/bash

# download scripts for Spec-RL
# 创建目录
mkdir -p data/deepmath data/simplerl_math_35

# 下载 deepmath
wget -O data/deepmath/train_sample_6144.parquet \
  https://huggingface.co/datasets/bensonbsliu/Spec-RL/resolve/main/deepmath/train_sample_6144.parquet

wget -O data/deepmath/test.parquet \
  https://huggingface.co/datasets/bensonbsliu/Spec-RL/resolve/main/deepmath/test.parquet

# 下载 simplerl_math_35
wget -O data/simplerl_math_35/train_8192.parquet \
  https://huggingface.co/datasets/bensonbsliu/Spec-RL/resolve/main/simplerl_math_35/train_8192.parquet

wget -O data/simplerl_math_35/test.parquet \
  https://huggingface.co/datasets/bensonbsliu/Spec-RL/resolve/main/simplerl_math_35/test.parquet