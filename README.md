

Important.

Once you run pretraining and the models are saved you can use these for alternating and hierarchical control. More details provided below

GitHub link https://github.com/Vamsi-Bath/LifeLong-Metra for mostly full access to code still file size limitations on GitHub but should be able to run.

The CIFAR and MNIST dataset isn't provided but the code when run should automatically download the necessary files to run our experiments if not present

Code Accreditation

Metra has been used as the backbone for my lifelong Metra implementation and I used the original Metra Implementation but the code has been modified to suit my experiments and with gym deprecated  https://github.com/seohongpark/METRA

The continual backpropagation code for PPO and SAC has been used and also the permuted MNIST and Incremental CIFAR experiment ideas have been adopted but have been modified extensively to fit my experiments
https://github.com/shibhansh/loss-of-plasticity

My implementation of RND and PPO were borrowed from my dissertation last year which at the time was cited from my group reinforcement learning project: VAMSI https://github.com/pkr895/Reinforcement-Learning

For the hierarchical control PPO I have borrowed code from the contrastive successor features. I have used the child policy wrapper from
https://princeton-rl.github.io/contrastive-successor-features/
https://github.com/mturan33/isaac-g1-ulc  only the networks using tanh and CReLU have been borrowed here the rest of this PPO implementation is taken from my RND implementation

On top of my RND implementation, we also used the RLeXplore RND baseline code used in the METRA paper https://github.com/RLE-Foundation/RLeXplore


## Installation

```
conda create --name metra python=3.8
conda activate metra
pip install -r requirements.txt --no-deps
pip install -e .
pip install -e garaged
```
For everything to run, cd to METRA

Supervised Learning experiments:

For CIFAR experiment run python lop\incremental_cifar\run_cifar.py
config file to modify at wish lop\incremental_cifar\tempCIFAR_cfg\cifar_cbp_compare.json

For the PERMUTED MNIST experiment run python lop\permuted_mnist\cbp_compare.py
config file to modify at wish lop\permuted_mnist\cfg\cbp_compare.json

RL experiments   
RND python -m iod.RND.TrainPPOAgent --env_name Humanoid-v5 --Network Dense --action_type continuous --nonstat 1 --nonstat_type Hard --var_target mass --base_var 8 --delta_var 0.5 --omegaVar 0.3 --interval 200 --var_bodies torso
n_epochs[ x ,  y  ] choose x for ant and choose y for humanoid       everything else [,] choose a value   

For the hierarchical_control and alternating modes, you need --cp_path, pass in the relative path of the child policy folder which are your pretraining models which are stored in METRA\exp\pretraining  
This will only work if you trained your pretraining agent past the save intervals all set to 200 epochs where option_policyXXXX.pt will be generated and is required in the folder

Due to size limitations only managed to provide 1 for submission
you can use this as an example
                                  METRA\exp\pretraining\sd000_1776265950_ant_metra_lifelong_ctx2
 
METRA\exp\pretraining\sd000_1776263709_ant_metra 
1st is METRA
2nd is Lifelong METRA

python tests/main.py --mode pretrain --pretrain_algo metra --dim_option 4 --n_epochs 1500 --cbp [0,1] --env ['ant', 'humanoid']
python tests/main.py --mode pretrain --pretrain_algo metra --lifelong 1 -- dim_option 2 --context_dim 2 --n_epochs 1500 --cbp [0,1] --env ['ant', 'humanoid']
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
python tests/main.py --mode hierarchical --pretrain_algo metra --dim_option 4 --n_epochs [1500,3000] --cbp [0,1] --env ['ant', 'humanoid'] --hierarchical_algo 'ppo' --cp_path METRA\exp\pretraining\sd000_1776263709_ant_metra

python tests/main.py --mode hierarchical --pretrain_algo metra --lifelong 1 -- dim_option 2 --context_dim 2 --n_epochs [1500,3000] --cbp [0,1] --env ['ant', 'humanoid'] --hierarchical_algo 'ppo' --cp_path METRA\exp\pretraining\sd000_1776265950_ant_metra_lifelong_ctx2
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
python tests/main.py --mode alternating --pretrain_algo metra --dim_option 4 --n_epochs 1500 --cbp [0,1] --env 'ant'--hierarchical_algo 'ppo' --cp_path METRA\exp\pretraining\sd000_1776263709_ant_metra 

python tests/main.py --mode alternating --pretrain_algo metra --lifelong 1 -- dim_option 2 --context_dim 2 --n_epochs 1500 --cbp [0,1] --env 'ant' --hierarchical_algo 'ppo' --cp_path METRA\exp\pretraining\sd000_1776265950_ant_metra_lifelong_ctx2
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
RND experiment:
python -m iod.RND.TrainPPOAgent --env_name Humanoid-v5 --Network Dense --action_type continuous --nonstat 1 --nonstat_type Hard --var_target friction
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Visualise Lifelong Metra MUJOCO Ant:

python -m tests.ViSkills --mode skills --algorithm lifelong_metra --cbp_mode NoCBP --pretraining_folder .\exp\pretraining\sd000_1776265950_ant_metra_lifelong_ctx2 --model_name option_policy900.pt


python -m tests.ViSkills --mode skills --algorithm lifelong_metra --cbp_mode CBP --pretraining_folder .\exp\pretraining\sd000_1776265950_ant_metra_lifelong_ctx2 --model_name option_policy900.pt

python -m tests.ViSkills --mode hierarchical --algorithm lifelong_metra --cbp_mode NoCBP --pretraining_folder .\exp\pretraining\sd000_1776265950_ant_metra_lifelong_ctx2 --model_name option_policy900.pt


LICENSES to Modify the MUJOCO Ant and Humanoid XML files are provided
LICENCE to use GARAGE is provided





