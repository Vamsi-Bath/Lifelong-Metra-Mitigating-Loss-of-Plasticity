#!/usr/bin/env python3
import dowel_wrapper
assert dowel_wrapper is not None
import argparse
import os
import sys
import platform
import torch.multiprocessing as mp
import gymnasium as gym
import better_exceptions
import numpy as np

from tests.loader.AntTest.test_agent import testPretrain, testHierarhical, testRND

plat = platform.platform().lower()
if 'mac' in plat:
    pass
elif sys.platform.startswith('win'):
    os.environ['MUJOCO_GL'] = 'glfw'
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

better_exceptions.hook()

EXP_DIR = 'exp'
START_METHOD = os.environ.get('START_METHOD', 'spawn')


def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="lifelong_metra",
        choices=["metra", "lifelong_metra", "dads", "ppo", "rnd"],
        help="Algorithm name passed to the selected test loader",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="skills",
        choices=["skills", "hierarchical", "rnd"],
        help="Which demo to run",
    )
    parser.add_argument(
        "--cbp_mode",
        type=str,
        default="NoCBP",
        choices=["CBP", "NoCBP"],
        help="Select whether to load from the CBP or NoCBP folder",
    )
    parser.add_argument(
        "--dim_option",
        type=int,
        default=2,
        help="Skill dimension for random_skill sampling",
    )
    parser.add_argument(
        "--dim_context",
        type=int,
        default=0,
        help="Context dimension",
    )
    parser.add_argument(
        "--pretraining_folder",
        type=str,
        default=None,
        help="Optional pretraining folder path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional model name. Leave unset for no fixed model name.",
    )
    return parser

def runSkills(args):
    print("=" * 60)

    algo_name = args.algorithm
    pretraining_folder = args.pretraining_folder
    model_name = args.model_name
    cbp_mode = args.cbp_mode

    # ---- set dimensions based on algorithm ----
    if algo_name == "lifelong_metra":
        dim_skill = 2
        dim_context = 2
    else:
        dim_skill = 4
        dim_context = 0

    # ---- sample skill ----
    skill = np.random.uniform(-1.0, 1.0, size=(dim_skill,)).astype(np.float32)

    if dim_context > 0:
        context = np.random.uniform(-1.0, 1.0, size=(dim_context,)).astype(np.float32)
        random_skill = np.concatenate([skill, context])
    else:
        random_skill = skill

    print("Selected mode: skills")
    print(f"Selected algorithm: {algo_name}")
    print(f"Selected CBP mode: {cbp_mode}")
    print(f"model_name: {model_name}")
    print(f"Skill dim: {dim_skill}, Context dim: {dim_context}")
    print(f"Sampled skill: {random_skill}")

    env = 'Ant'
    testPretrain(
        algorithm=algo_name,
        env=env,
        model_name=model_name,
        random_skill=random_skill,
        cbp_mode=cbp_mode,
    )


def runHierarhical(args):
    print("\n" + "=" * 60)
    print("[VISUALISE SKILLS] hierarchical demo mode")
    print("=" * 60)

    algo_name = args.algorithm
    pretraining_folder = args.pretraining_folder
    model_name = args.model_name
    cbp_mode = args.cbp_mode

    print("Selected mode: hierarchical")
    print(f"Selected algorithm: {algo_name}")
    print(f"Selected CBP mode: {cbp_mode}")
    print(f"pretraining_folder: {pretraining_folder}")
    print(f"model_name: {model_name}")

    env = 'Ant'
    testHierarhical(
        algorithm=algo_name,
        env=env,
        model_name=model_name,
        cbp_mode=cbp_mode,
    )


if __name__ == "__main__":
    args = get_argparser().parse_args()

    if args.mode == "skills":
        runSkills(args)
    elif args.mode == "hierarchical":
        runHierarhical(args)
    else:
        env = 'Ant'
        testRND(
            env=env,
            model_name=args.model_name,
        )