<div align="center">
  <img src="./assets/page.jpg" alt="Logo" width="500">
</div>

# SPEC-RL: Accelerating On-Policy Reinforcement Learning via Speculative Rollouts


<div align="center">
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-CC%20BY%204.0-green.svg" alt="License">
<img src="https://img.shields.io/github/stars/ShopeeLLM/Spec-RL?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/ShopeeLLM/Spec-RL?color=red" alt="Issues">
<img src="https://img.shields.io/badge/python-3.8-purple.svg" alt="Python">



**_¹ <sup>+</sup> [Bingshuai Liu](https://bingshuailiu.github.io), ¹ ³ <sup>+</sup> Ante Wang, ¹ <sup>+</sup> Zijun Min,_**

**_² Liang Yao, ² Haibo Zhang, ³ [Yang Liu](https://nlp.csai.tsinghua.edu.cn/~ly/), ² [Anxiang Zeng](https://sites.google.com/view/anxiang-zeng/home), ¹ <sup>*</sup>[Jinsong Su](https://cdmc.xmu.edu.cn/info/1010/1054.htm)_**


_¹ Xiamen University, ² Shopee LLM Team, ³ Institute for AI Industry Research (AIR), Tsinghua University_

_<sup>+</sup> Equal Contribution_
_<sup>*</sup>Jinsong Su is the corresponding author: [jssu@xmu.edu.cn](mailto:jssu@xmu.edu.cn)_
</div>



## Installation

Use **either** Docker (fastest) **or** a Conda environment aligned with **verl 0.5.x**.

### Option A — Docker (recommended)

Use this prebuilt image (no further steps here):

**`verlai/verl:app-verl0.5-vllm0.9.1-mcore0.12.2-te2.2`**

### Option B — Conda (VERL 0.5.x)

Follow verl's official 0.5.x installation guide to set up the environment (PyTorch, vLLM, etc.):

https://verl.readthedocs.io/en/v0.5.x/start/install.html#install-dependencies


## Training
This repo ships two shell scripts under `training_scripts/`. Please **download datasets first**, then **configure paths**, then **launch**.

### 1) Download code

````
git clone https://github.com/ShopeeLLM/Spec-RL
````

### 2) Download datasets

```bash
bash data/download.sh
````

This will populate `data/` with the required files.

### 3) Configure the scripts

There are **two** scripts to edit before running:

#### (A) Vanilla GRPO baseline

1. Set **your own project root**:

   File: `training_scripts/vanilla-grpo/1.7B-grpo.sh`

```bash
# inside training_scripts/vanilla-grpo/1.7B-grpo.sh
PROJECT_DIR=path-to-root-project-dir
MODEL_PATH=path-to-your-base-model-path
SAVE_PATH=path-to-your-save-path
PROJECT_NAME=your-custom-project-name
```

2. Set GRPO main script:

File: `training_scripts/train_grpo.sh`

Set the following variables to **your own paths**:

```bash
# inside training_scripts/train_grpo.sh
MODEL_PATH=path-to-default-base-model-dir
CHECKPOINT_PATH=path-to-default-save-model-path
```

#### (B) SPEC-RL

1. Set **your own project root** and **SPEC-RL parameters ⚡**:

   File: `training_scripts/spec-rl/1.7B-grpo-lenience-0.5.sh`

```bash
# inside training_scripts/spec-rl/1.7B-grpo-lenience-0.5.sh
PROJECT_DIR=path-to-root-project-dir
MODEL_PATH=path-to-your-base-model-path
SAVE_PATH=path-to-your-save-path
PROJECT_NAME=your-custom-project-name

# turn on speculative decoding mode and lenience
--spec_decoding True \
--bias 0.5 \
```

2. Set specl-rl GRPO main script:

File: `training_scripts/train_grpo-spec-sampling.sh`

Set the following variables to **your own paths** and default **SPEC-RL parameters** :

```bash
# inside training_scripts/train_grpo-spec-sampling.sh
MODEL_PATH=path-to-default-base-model-dir
CHECKPOINT_PATH=path-to-default-save-model-path

SPEC_DECODING=False
BIAS=0.0
```

### 3) Login to Weights & Biases

```bash
wandb login
```

### 4) Launch training

**Vanilla GRPO baseline** (recommended first run):

```bash
bash training_scripts/vanilla-grpo/1.7B-grpo.sh
```

**SPEC-RL (with speculative decoding) GRPO**:

```bash
bash training_scripts/spec-rl/1.7B-grpo-lenience-0.5.sh
```

After the first run, monitor logs under `logs/` (and your W\&B project if enabled).

