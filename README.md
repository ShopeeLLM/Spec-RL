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



**_¹ [Bingshuai Liu](https://bingshuailiu.github.io), ¹ Ante Wang, ¹ Zijun Min,_**

**_² Liang Yao, ² Haibo Zhang, ² Anxiang Zeng, ¹ <sup>*</sup>Jinsong Su_**


_¹ Xiamen University, ² Shopee LLM Team_

_<sup>*</sup>Jinsong is the corresponding author: [jssu@xmu.edu.cn](mailto:jssu@xmu.edu.cn)_
</div>



## Installation

Use **either** Docker (fastest) **or** a Conda environment aligned with **verl 0.5.x**.

---

### Option A — Docker (recommended)

Use this prebuilt image (no further steps here):

**`verlai/verl:app-verl0.5-vllm0.9.1-mcore0.12.2-te2.2`**

---

### Option B — Conda (VERL 0.5.x)

Follow verl's official 0.5.x installation guide to set up the environment (PyTorch, vLLM, etc.):

https://verl.readthedocs.io/en/v0.5.x/start/install.html#install-from-custom-environment


---


## Training
This repo ships two shell scripts under `training_scripts/`. Please **download datasets first**, then **configure paths**, then **launch**.

---

### 1) Download code

````
git clone https://github.com/ShopeeLLM/Spec-RL
````

### 2) Download datasets

```bash
bash data/download.sh
````

This will populate `data/` with the required files.

---

### 3) Configure the scripts

There are **two** scripts to edit before running:

#### (A) Vanilla GRPO baseline

1. Set **your own project root**:

   File: `${project_root_path}/training_scripts/vanilla-grpo/1.7B_grpo.sh`

```bash
# inside training_scripts/vanilla-grpo/1.7B_grpo.sh
PROJECT_DIR=path-to-root-project-dir
MODEL_PATH=path-to-your-base-model-path
SAVE_PATH=path-to-your-save-path
PROJECT_NAME=your-custom-project-name
```

2. Set GRPO main script:

File: `${project_root_path}/training_scripts/train_grpo.sh`

Set the following variables to **your own paths**:

```bash
# inside training_scripts/train_grpo.sh
MODEL_PATH=path-to-default-base-model-dir
CHECKPOINT_PATH=path-to-default-save-model-path
```

---

#### (B) SPEC-RL

1. Set **your own project root** and **SPEC-RL parameters ⚡**:

   File: `${project_root_path}/training_scripts/spec-rl/1.7B-grpo-lenience-0.5.sh`

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

File: `${project_root_path}/training_scripts/train_grpo-spec-sampling.sh`

Set the following variables to **your own paths** and default **SPEC-RL parameters** :

```bash
# inside training_scripts/train_grpo.sh
MODEL_PATH=path-to-default-base-model-dir
CHECKPOINT_PATH=path-to-default-save-model-path

SPEC_DECODING=False
BIAS=0.0
```

---



### 3) Login to Weights & Biases

```bash
wandb login
```

---

### 4) Launch training

**Vanilla GRPO baseline** (recommended first run):

```bash
# use the exact filename present in your repo
bash training_scripts/vanilla-grpo/1.7B_grpo.sh
```

**SPEC-RL (with speculative decoding) GRPO**:

```bash
bash training_scripts/spec-rl/1.7B-grpo-lenience-0.5.sh
```

After the first run, monitor logs under `logs/` (and your W\&B project if enabled).

---