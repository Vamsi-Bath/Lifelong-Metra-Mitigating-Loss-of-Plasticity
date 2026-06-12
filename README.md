# METRA Lifelong Learning Experiments

This repository contains experiments for supervised continual learning, reinforcement learning, lifelong METRA, hierarchical control, alternating control, and RND-based exploration.

The CIFAR and MNIST datasets are **not included** in this repository. When the relevant scripts are run, the required dataset files should be downloaded automatically if they are not already present.

---

## Code Accreditation

This project builds on and modifies several existing codebases.

### METRA

METRA has been used as the backbone for my lifelong METRA implementation. I used the original METRA implementation and modified it to suit my experiments. The code has also been adapted because `gym` is deprecated.

Original METRA implementation:

https://github.com/seohongpark/METRA

### Continual Backpropagation, PPO, SAC, Permuted MNIST, and Incremental CIFAR

The continual backpropagation code for PPO and SAC has been used. The Permuted MNIST and Incremental CIFAR experiment ideas were also adopted and extensively modified to fit my experiments.

Original repository:

https://github.com/shibhansh/loss-of-plasticity

### RND and PPO

My implementation of RND and PPO was borrowed from my dissertation project from last year. At the time, this was cited from my group reinforcement learning project:

VAMSI:

https://github.com/pkr895/Reinforcement-Learning

### Hierarchical Control PPO

For hierarchical control PPO, I borrowed code from Contrastive Successor Features. I used the child policy wrapper from:

https://princeton-rl.github.io/contrastive-successor-features/

I also borrowed only the networks using `tanh` and `CReLU` from:

https://github.com/mturan33/isaac-g1-ulc

The rest of this PPO implementation is taken from my RND implementation.

### RLeXplore RND Baseline

On top of my RND implementation, I also used the RLeXplore RND baseline code used in the METRA paper:

https://github.com/RLE-Foundation/RLeXplore

---

## Installation

Create and activate the Conda environment:

```bash
conda create --name metra python=3.8
conda activate metra
```

Install the required dependencies:

```bash
pip install -r requirements.txt --no-deps
pip install -e .
pip install -e garaged
```

For everything to run correctly, make sure you are inside the `METRA` directory:

```bash
cd METRA
```

---

## Supervised Learning Experiments

### Incremental CIFAR

Run:

```bash
python lop/incremental_cifar/run_cifar.py
```

Configuration file:

```text
lop/incremental_cifar/tempCIFAR_cfg/cifar_cbp_compare.json
```

### Permuted MNIST

Run:

```bash
python lop/permuted_mnist/cbp_compare.py
```

Configuration file:

```text
lop/permuted_mnist/cfg/cbp_compare.json
```

---

## Reinforcement Learning Experiments

### RND PPO Experiment

Example command:

```bash
python -m iod.RND.TrainPPOAgent \
  --env_name Humanoid-v5 \
  --Network Dense \
  --action_type continuous \
  --nonstat 1 \
  --nonstat_type Hard \
  --var_target mass \
  --base_var 8 \
  --delta_var 0.5 \
  --omegaVar 0.3 \
  --interval 200 \
  --var_bodies torso
```

For `n_epochs`, choose the appropriate value depending on the environment:

* Choose the first value for Ant.
* Choose the second value for Humanoid.

For all other list-style arguments, choose one value from the list.

---

## METRA Pretraining Experiments

### Standard METRA Pretraining

```bash
python tests/main.py \
  --mode pretrain \
  --pretrain_algo metra \
  --dim_option 4 \
  --n_epochs 1500 \
  --cbp 0 \
  --env ant
```

To run with CBP enabled:

```bash
python tests/main.py \
  --mode pretrain \
  --pretrain_algo metra \
  --dim_option 4 \
  --n_epochs 1500 \
  --cbp 1 \
  --env ant
```

For Humanoid, replace `ant` with `humanoid`:

```bash
python tests/main.py \
  --mode pretrain \
  --pretrain_algo metra \
  --dim_option 4 \
  --n_epochs 1500 \
  --cbp 0 \
  --env humanoid
```

### Lifelong METRA Pretraining

```bash
python tests/main.py \
  --mode pretrain \
  --pretrain_algo metra \
  --lifelong 1 \
  --dim_option 2 \
  --context_dim 2 \
  --n_epochs 1500 \
  --cbp 0 \
  --env ant
```

To run with CBP enabled:

```bash
python tests/main.py \
  --mode pretrain \
  --pretrain_algo metra \
  --lifelong 1 \
  --dim_option 2 \
  --context_dim 2 \
  --n_epochs 1500 \
  --cbp 1 \
  --env ant
```

For Humanoid, replace `ant` with `humanoid`.

---

## Hierarchical Control Experiments

For hierarchical control and alternating modes, you need to pass `--cp_path`.

The `--cp_path` argument should be the relative path to the child policy folder. These folders contain pretrained models and are stored in:

```text
METRA/exp/pretraining
```

This will only work if the pretraining agent was trained past the save interval. The save interval is set to 200 epochs, and files such as the following must exist in the folder:

```text
option_policyXXXX.pt
```

Example pretraining folders:

```text
METRA/exp/pretraining/sd000_1776263709_ant_metra
METRA/exp/pretraining/sd000_1776265950_ant_metra_lifelong_ctx2
```

The first folder is for standard METRA.

The second folder is for Lifelong METRA.

---

## Hierarchical PPO with Standard METRA

```bash
python tests/main.py \
  --mode hierarchical \
  --pretrain_algo metra \
  --dim_option 4 \
  --n_epochs 1500 \
  --cbp 0 \
  --env ant \
  --hierarchical_algo ppo \
  --cp_path METRA/exp/pretraining/sd000_1776263709_ant_metra
```

To run with CBP enabled:

```bash
python tests/main.py \
  --mode hierarchical \
  --pretrain_algo metra \
  --dim_option 4 \
  --n_epochs 1500 \
  --cbp 1 \
  --env ant \
  --hierarchical_algo ppo \
  --cp_path METRA/exp/pretraining/sd000_1776263709_ant_metra
```

For longer training, use:

```bash
python tests/main.py \
  --mode hierarchical \
  --pretrain_algo metra \
  --dim_option 4 \
  --n_epochs 3000 \
  --cbp 0 \
  --env ant \
  --hierarchical_algo ppo \
  --cp_path METRA/exp/pretraining/sd000_1776263709_ant_metra
```

For Humanoid, replace `ant` with `humanoid` and provide the correct Humanoid pretraining folder.

---

## Hierarchical PPO with Lifelong METRA

```bash
python tests/main.py \
  --mode hierarchical \
  --pretrain_algo metra \
  --lifelong 1 \
  --dim_option 2 \
  --context_dim 2 \
  --n_epochs 1500 \
  --cbp 0 \
  --env ant \
  --hierarchical_algo ppo \
  --cp_path METRA/exp/pretraining/sd000_1776265950_ant_metra_lifelong_ctx2
```

To run with CBP enabled:

```bash
python tests/main.py \
  --mode hierarchical \
  --pretrain_algo metra \
  --lifelong 1 \
  --dim_option 2 \
  --context_dim 2 \
  --n_epochs 1500 \
  --cbp 1 \
  --env ant \
  --hierarchical_algo ppo \
  --cp_path METRA/exp/pretraining/sd000_1776265950_ant_metra_lifelong_ctx2
```

For longer training, use:

```bash
python tests/main.py \
  --mode hierarchical \
  --pretrain_algo metra \
  --lifelong 1 \
  --dim_option 2 \
  --context_dim 2 \
  --n_epochs 3000 \
  --cbp 0 \
  --env ant \
  --hierarchical_algo ppo \
  --cp_path METRA/exp/pretraining/sd000_1776265950_ant_metra_lifelong_ctx2
```

For Humanoid, replace `ant` with `humanoid` and provide the correct Humanoid pretraining folder.

---

## Alternating Control Experiments

### Alternating Control with Standard METRA

```bash
python tests/main.py \
  --mode alternating \
  --pretrain_algo metra \
  --dim_option 4 \
  --n_epochs 1500 \
  --cbp 0 \
  --env ant \
  --hierarchical_algo ppo \
  --cp_path METRA/exp/pretraining/sd000_1776263709_ant_metra
```

To run with CBP enabled:

```bash
python tests/main.py \
  --mode alternating \
  --pretrain_algo metra \
  --dim_option 4 \
  --n_epochs 1500 \
  --cbp 1 \
  --env ant \
  --hierarchical_algo ppo \
  --cp_path METRA/exp/pretraining/sd000_1776263709_ant_metra
```

### Alternating Control with Lifelong METRA

```bash
python tests/main.py \
  --mode alternating \
  --pretrain_algo metra \
  --lifelong 1 \
  --dim_option 2 \
  --context_dim 2 \
  --n_epochs 1500 \
  --cbp 0 \
  --env ant \
  --hierarchical_algo ppo \
  --cp_path METRA/exp/pretraining/sd000_1776265950_ant_metra_lifelong_ctx2
```

To run with CBP enabled:

```bash
python tests/main.py \
  --mode alternating \
  --pretrain_algo metra \
  --lifelong 1 \
  --dim_option 2 \
  --context_dim 2 \
  --n_epochs 1500 \
  --cbp 1 \
  --env ant \
  --hierarchical_algo ppo \
  --cp_path METRA/exp/pretraining/sd000_1776265950_ant_metra_lifelong_ctx2
```

---

## Additional RND Experiment

To run RND with a friction-based non-stationary environment:

```bash
python -m iod.RND.TrainPPOAgent \
  --env_name Humanoid-v5 \
  --Network Dense \
  --action_type continuous \
  --nonstat 1 \
  --nonstat_type Hard \
  --var_target friction
```

---

## Visualising Lifelong METRA on MuJoCo Ant

### Visualise Lifelong METRA Skills without CBP

```bash
python -m tests.ViSkills \
  --mode skills \
  --algorithm lifelong_metra \
  --cbp_mode NoCBP \
  --pretraining_folder ./exp/pretraining/sd000_1776265950_ant_metra_lifelong_ctx2 \
  --model_name option_policy900.pt
```

### Visualise Lifelong METRA Skills with CBP

```bash
python -m tests.ViSkills \
  --mode skills \
  --algorithm lifelong_metra \
  --cbp_mode CBP \
  --pretraining_folder ./exp/pretraining/sd000_1776265950_ant_metra_lifelong_ctx2 \
  --model_name option_policy900.pt
```

### Visualise Lifelong METRA Hierarchical Control without CBP

```bash
python -m tests.ViSkills \
  --mode hierarchical \
  --algorithm lifelong_metra \
  --cbp_mode NoCBP \
  --pretraining_folder ./exp/pretraining/sd000_1776265950_ant_metra_lifelong_ctx2 \
  --model_name option_policy900.pt
```

---

## Notes

Use forward slashes `/` in paths for better compatibility across platforms.

For example, use:

```text
METRA/exp/pretraining/sd000_1776263709_ant_metra
```

instead of:

```text
METRA\exp\pretraining\sd000_1776263709_ant_metra
```

On Windows, both may work depending on the shell, but forward slashes are usually safer in Markdown documentation.

---

## Licenses

Licenses to modify the MuJoCo Ant and Humanoid XML files are provided.

The license to use GARAGE is also provided.
