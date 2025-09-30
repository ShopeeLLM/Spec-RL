<div align="center">
  <img src="./assets/page.jpg" alt="Logo" width="500">
</div>

# SPEC-RL: Accelerating On-Policy Reinforcement Learning via Speculative Rollouts

<div align="center">
  
[![Project Page](https://img.shields.io/badge/🌐-Project%20Page-brightgreen.svg)](https://bingshuailiu.github.io/Spec-RL/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.23232-b31b1b.svg)](https://arxiv.org/abs/2509.23232)
<img src="https://img.shields.io/badge/License-CC%20BY%204.0-green.svg" alt="License">
<img src="https://img.shields.io/github/stars/ShopeeLLM/Spec-RL?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/ShopeeLLM/Spec-RL?color=red" alt="Issues">
[![📊 WandB Report](https://img.shields.io/badge/WandB-Report-orange.svg?logo=weightsandbiases&logoColor=white)](https://wandb.ai/bingshuai-liu/spec-rl/reports/SPEC-RL-Report--VmlldzoxNDQ1MDE0OQ?accessToken=wlggqshgos8pbp7r7s6h2ib9onh9woplphhqd2tf6dx8fcw00t4fdmtpvl3aeii2)



**_¹ <sup>+</sup> [Bingshuai Liu](https://bingshuailiu.github.io), ¹ ³ <sup>+</sup> [Ante Wang](https://soistesimmer.github.io/), ¹ <sup>+</sup> Zijun Min,_**

**_² Liang Yao, ² Haibo Zhang, ³ [Yang Liu](https://nlp.csai.tsinghua.edu.cn/~ly/), ² [Anxiang Zeng](https://sites.google.com/view/anxiang-zeng/home), ¹ <sup>*</sup>[Jinsong Su](https://cdmc.xmu.edu.cn/info/1010/1054.htm)_**


_¹ Xiamen University, ² Shopee LLM Team, ³ Institute for AI Industry Research (AIR), Tsinghua University_

_<sup>+</sup> Equal Contribution_
_<sup>*</sup>Jinsong Su is the corresponding author: [jssu@xmu.edu.cn](mailto:jssu@xmu.edu.cn)_
</div>

## 📢 News

Stay tuned for our latest updates and milestones!

- **[2025-09-30]** 📊 **Full Results Published on WandB**  
  We published the full set of experiment results as a **WandB report**.  
  🔗 [View the report on WandB](https://wandb.ai/bingshuai-liu/spec-rl/reports/SPEC-RL-Report--VmlldzoxNDQ1MDE0OQ?accessToken=wlggqshgos8pbp7r7s6h2ib9onh9woplphhqd2tf6dx8fcw00t4fdmtpvl3aeii2)

- **[2025-09-28]** 🎉 **SPEC-RL Release!**  
  Official release of SPEC-RL with 2–3× rollout acceleration and seamless integration into PPO, GRPO, and DAPO workflows. 

## 🧩 Installation

Use **either** Docker (fastest) **or** a Conda environment aligned with **verl 0.5.x**.

### Option A — Docker (recommended)

Use this prebuilt image (no further steps here):

**`verlai/verl:app-verl0.5-vllm0.9.1-mcore0.12.2-te2.2`**

### Option B — Conda (VERL 0.5.x)

Follow verl's official 0.5.x installation guide to set up the environment (PyTorch, vLLM, etc.):

https://verl.readthedocs.io/en/v0.5.x/start/install.html#install-dependencies


## 🚀 Training
This repo ships two shell scripts under `training_scripts/`. Please **download datasets first**, then **configure paths**, then **launch**.

### 1️⃣ Download code

````
git clone https://github.com/ShopeeLLM/Spec-RL
````

### 2️⃣ Download datasets

```bash
bash data/download.sh
````

This will populate `data/` with the required files.

### 3️⃣ Configure the scripts

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

### 4️⃣ Login to Weights & Biases

```bash
wandb login
```

### 5️⃣ Launch training

**Vanilla GRPO baseline** (recommended first run):

```bash
bash training_scripts/vanilla-grpo/1.7B-grpo.sh
```

**SPEC-RL (with speculative decoding) GRPO**:

```bash
bash training_scripts/spec-rl/1.7B-grpo-lenience-0.5.sh
```

After the first run, monitor logs under `logs/` (and your W\&B project if enabled).


## 🧠 Evaluation

We provide evaluation scripts adapted from **[SimpleRL-Reason](https://github.com/hkust-nlp/simpleRL-reason)** for math reasoning benchmarks.

### 1️⃣ Configure the scripts

There are **two** files to edit before running:

#### (A) eval/example.sh

Configure these three paths:

```bash
PROJECT_DIR=path-to-your-base-project-dir
CKPT_DIR=path-to-your-ckpt-to-be-evaluated
BASE_MODEL_DIR=path-to-your-base-model
```

#### (B) eval/eval_math_nodes.sh

Configure this variable:

```bash
PROJECT_DIR=path-to-your-base-project-dir
```

---

### 2️⃣ Run evaluation

After configuration, directly run:

```bash
bash eval/example.sh
```

This will automatically:
- enter the evaluation directory  
- install required dependencies  
- and launch evaluation across multiple benchmarks


------

## 📘 Citation
If you find **SPEC-RL** helpful, please cite:
```bibtex
@misc{spec_rl_2025,
  title         = {SPEC-RL: Accelerating On-Policy Reinforcement Learning via Speculative Rollouts},
  author        = {Liu, Bingshuai and Wang, Ante and Min, Zijun and Yao, Liang and Zhang, Haibo and Liu, Yang and Zeng, Anxiang and Su, Jinsong},
  year          = {2025},
  howpublished  = {\url{https://github.com/ShopeeLLM/Spec-RL}},
  note          = {Accessed: 2025-09-28}
}
```
