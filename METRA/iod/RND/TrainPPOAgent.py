import gymnasium as gym
from iod.RND.PPOAgent import Agent
from iod.RND import EnvironmentWrappers
from iod.RND.EnvironmentWrappers import LogReturnAndEntropy
from iod.NonStatUpdate import apply_nonstationary_wrapper
from iod.RND.logger import CSVLoggerTest, CSVLoggerTrain
import numpy as np
import argparse
import sys
import os
from gymnasium.wrappers import RecordVideo
import torch as T
import ale_py
import wandb
import time
import random
import math
import gymnasium_robotics

def process_observation(obs):
    if isinstance(obs, dict):
        flat_parts = []
        for k in sorted(obs.keys()):
            v = obs[k]
            if isinstance(v, np.ndarray):
                flat_parts.append(v.astype(np.float32).flatten())
            else:
                flat_parts.append(np.array(v, dtype=np.float32).flatten())
        return np.concatenate(flat_parts)
    else:
        return np.atleast_1d(np.array(obs, dtype=np.float32))

def initialize_wandb(project_name, run_name, config):
    if wandb.run is None:
        wandb.init(
            project=project_name,
            name=run_name,
            config=config
        )
        print(f"WandB initialized with project: {project_name}, run: {run_name}")
    else:
        print("WandB is already initialized. Skipping reinitialization.")
        
def create_pyenv(env_name="SpaceInvadersDeterministic-v4", render_mode="rgb_array", Network='Dense'):
    env = gym.make(env_name, render_mode=render_mode)
    initial_obsShape = env.observation_space.shape
    if Network == "CNN":
        env = EnvironmentWrappers.NoopResetEnv(env, noop_max=30)
        env = EnvironmentWrappers.MaxAndSkipEnv(env, skip=4)
        env = EnvironmentWrappers.EpisodicLifeEnv(env)
        env = EnvironmentWrappers.FireResetEnv(env)
        env = EnvironmentWrappers.WarpFrame(env)
        env = EnvironmentWrappers.ScaledFloatFrame(env)
        env = EnvironmentWrappers.PyTorchFrame(env)
        env = EnvironmentWrappers.FrameStack(env, 4)
    return env, initial_obsShape

def train_agent(agent, env, seed, Iters, UpdateFrequency, save_interval, Record_Video, Rolling_Average):
    scores = []

    logName = agent.chkpt_dir
    csv_logger = CSVLoggerTrain(logName)
    video_folder = './BestScores'
    avg_score, best_avg_score, best_episodic_score = -np.inf, -np.inf, -np.inf
    avg_history, n_steps, learn_iters = [], 0, 0
    env.reset(seed=seed)

    if os.path.exists(agent.actor.checkpoint_file) and os.path.exists(agent.critic.checkpoint_file):
        print("Loading existing models")
        agent.load_models()
    else:
        print("No saved models found. Starting fresh.")

    intrinsic_rewards_per_update = []
    for i in range(Iters):
        if Record_Video:
            env = RecordVideo(env, video_folder="./recorded_videos", episode_trigger=lambda ep_id: ep_id == i)
        obs, _ = env.reset()
        done, truncated = False, False
        iter_score = 0
        episodicIntrinsicReward = 0
        time.sleep(0.5)
        

        while not (done or truncated):

            obs = process_observation(obs)
            action, prob, val, entropy = agent.choose_action(obs)
            obs_, extrinsic_reward, done, truncated, info = env.step(action)
            intrinsic_reward = 0
            if agent.RND:
                intrinsic_reward = agent.rnd.calculate_intrinsic_reward(process_observation(obs_)) / agent.Reward_Scaler
            episodicIntrinsicReward += intrinsic_reward
    
            total_reward = extrinsic_reward + intrinsic_reward
            agent.store_transition(obs, action, prob, val, extrinsic_reward, intrinsic_reward, done, truncated)
            total_loss, actor_losses, critic_losses = 0 , [] , []

            iter_score += total_reward
            n_steps += 1

            if n_steps % UpdateFrequency == 0:
                total_loss,actor_losses, critic_losses = agent.learn()
                learn_iters += 1
                if agent.RND:
                    avg_intrinsic_reward = episodicIntrinsicReward / n_steps
                    intrinsic_rewards_per_update.append(avg_intrinsic_reward)

            csv_logger.log(
                        episode = i,
                        timestep = n_steps,
                        state=obs,
                        action=action,
                        extrinsic_reward=extrinsic_reward,
                        intrinsic_reward=intrinsic_reward,
                        total_reward=total_reward,
                        next_state = obs_,
                        entropy=entropy,
                        learn_step=learn_iters,
                        total_loss = total_loss,
                        actor_losses = actor_losses,
                        critic_losses = critic_losses
                    )
            obs = obs_ 

        if iter_score > best_episodic_score:
            best_episodic_score = iter_score
        
        iter_score += args.RNDRewardScaler
        rewards = [iter_score, episodicIntrinsicReward]
        LogReturnAndEntropy(agent.c2, rewards, csv_logger.file_path,agent.RND)

        if Record_Video:
            print(f"New best score: {best_episodic_score:.1f}. Saving video for episode {i}.")
            src_video_path = os.path.join("./recorded_videos", f"rl-video-episode-{i}.mp4")
            dst_video_path = os.path.join(video_folder, f"best-video-episode-{i}.mp4")
            if os.path.exists(src_video_path):
                os.rename(src_video_path, dst_video_path)

        episodic_score = rewards[0]
        episodicExtrinsicReward = episodic_score - episodicIntrinsicReward
        avg_history.append(episodicExtrinsicReward)
        avg_score = np.mean(avg_history[-Rolling_Average:])

        if avg_score > best_avg_score:
            best_avg_score = avg_score
            agent.save_models()

        if i % save_interval == 0:
            print(f"Saving checkpoint at episode {i+1}")
            agent.save_models()

        print(f'Episode {i+1}, Episodic Score: {episodic_score:.2f}, Extrinsic reward: {episodicExtrinsicReward:.2f},'
              f' Intrinsic reward: {episodicIntrinsicReward}, Avg Score: {avg_score:.1f},'
              f' Total Steps: {n_steps}, Learning Steps: {learn_iters}')

def test_agent(agent, env, seed, Iters, Rolling_Average, Record_Video=False):

    scores = []
    video_folder = './BestTrainingScores'
    timesteps = 0

    csv_logger = CSVLoggerTest(logName=agent.chkpt_dir)

    avg_score, best_avg_score, best_episodic_score = -np.inf, -np.inf, -np.inf
    avg_history, n_steps, learn_iters = [], 0, 0
    env.reset(seed=seed)

    if os.path.exists(agent.actor.checkpoint_file) and os.path.exists(agent.critic.checkpoint_file):
        print("Loading existing models")
        agent.load_models()
    else:
        print("No saved models found. Starting fresh.")

    for i in range(Iters):
        if Record_Video:
            env = RecordVideo(env, video_folder="./recorded_videos", episode_trigger=lambda ep_id: ep_id == i)
        obs, _ = env.reset(seed=seed)
        done, truncated = False, False
        episodic_score = 0
        episodicIntrinsicReward = 0

        while not (done or truncated):
            timesteps += 1
            obs = process_observation(obs)
            action, prob, val, entropy = agent.choose_action(obs)
            obs_, extrinsic_reward, done, truncated, info = env.step(action)
            intrinsic_reward = 0
            if agent.RND:
                intrinsic_reward = agent.rnd.calculate_intrinsic_reward(obs_)
            episodicIntrinsicReward += intrinsic_reward

            total_reward = extrinsic_reward + intrinsic_reward

            csv_logger.log(
                episode = i,
                timestep = n_steps,
                state=obs,
                action=action,
                extrinsic_reward=extrinsic_reward,
                intrinsic_reward=intrinsic_reward,
                total_reward=total_reward,
                next_state = obs_,
            )
            agent.store_transition(obs, action, prob, val, extrinsic_reward, intrinsic_reward, done, truncated)
            obs = obs_
            episodic_score += total_reward
            n_steps += 1

        if episodic_score > best_episodic_score:
            best_episodic_score = episodic_score
        episodic_score = episodic_score
        episodicExtrinsicReward = episodic_score - episodicIntrinsicReward

        if agent.RND:
            episodic_score = episodicExtrinsicReward + episodicIntrinsicReward


        if Record_Video:
            print(f"New best score: {best_episodic_score:.1f}. Saving video for episode {i}.")
            src_video_path = os.path.join("./recorded_videos", f"rl-video-episode-{i}.mp4")
            dst_video_path = os.path.join(video_folder, f"best-video-episode-{i}.mp4")
            if os.path.exists(src_video_path):
                os.rename(src_video_path, dst_video_path)

        avg_history.append(episodicExtrinsicReward)
        avg_score = np.mean(avg_history[-Rolling_Average:])
        print(f'Episode {i+1}, Episodic Score: {episodic_score}, Extrinsic reward: {episodicExtrinsicReward},'
              f' Intrinsic reward: {episodicIntrinsicReward}, Avg Score: {avg_score:.1f},'
              f' Total Steps: {n_steps}, Learning Steps: {learn_iters}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PPO Agent with Custom Environment")
    
    parser.add_argument('--Variant', type=str, default='PPO Clip', help='Surrogate objective function: PPO Clip, PPO Penalty, or PPO No Clip')
    parser.add_argument('--TrainingIterations', type=int, default=50000, help='Number of training episodes')
    parser.add_argument('--TestingIterations', type=int, default=100, help='Number of testing episodes')
    parser.add_argument('--Record_Video', type=bool, default=False, help='Record videos during training/testing')
    parser.add_argument('--UpdateFrequency', type=int, default=50, help='Learning interval in steps')
    parser.add_argument('--c1', type=float, default=0.5, help='Value Function Coefficient')
    parser.add_argument('--c2', type=float, default=0.01, help='Entropy Coefficient')
    parser.add_argument('--batch_size', type=int, default=124, help='Batch size for training')
    parser.add_argument('--RollingAverage', type=int, default=100, help='Number of episodes to compute average score over')
    parser.add_argument('--KL_DIV', type=float, default=0.015, help='Initial KL divergence value')
    parser.add_argument('--RewardScaler', type=int, default=10, help='Scaler for rewards')
    parser.add_argument('--RNDRewardScaler', type=int, default=150, help='Scaler for RND rewards')
    parser.add_argument('--penaltyCoefficent', type=float, default=0.1, help='Initial Penalty Coefficient for PPO Penalty')
    parser.add_argument('--d_target', type=float, default=0.01, help='Target value for KL divergence')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs per learning update')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--RND', type=bool, default=True, help='Enable RND exploration bonus')
    parser.add_argument('--policy_clip', type=float, default=0.2, help='PPO policy clip range')
    parser.add_argument('--save_interval', type=int, default=100, help='Save model every n episodes')
    parser.add_argument('--env_name', type=str, default='Humanoid-v5', help='Environment name')

    parser.add_argument('--nonstat', type=bool, default=False, help='Use nonstationary environment')
    parser.add_argument('--nonstat_type', type=str, choices=['Hard', 'Soft'], default='Hard',
                    help='Type of nonstationarity (Hard or Soft)')
    
    parser.add_argument('--var_target', type=str, choices=['mass', 'friction'], default='mass',
                    help='What to vary: "mass" or "friction" (applied to var_bodies)')

    parser.add_argument('--base_var', type=float, default=8.38432,
                        help='Baseline variable value (m₀)')
    
    parser.add_argument('--delta_var', type=float, default=0.5,
                        help='Amplitude of variable variation (Δm)')
    
    parser.add_argument('--omegaVar', type=float, default=1/8*(math.pi),
                        help='Frequency of variable sinusoidal update (ω)')
    
    parser.add_argument('--var_bodies', nargs='+', default=['torso'],
                        help='List of body names to apply mass updates to')
    
    parser.add_argument('--interval', type=int, default=200,
                        help='Interval in timesteps at which to apply non-stationary updates')

    nonstat_suffix = ""
    if parser.get_default("nonstat"):
        nonstat_suffix = parser.get_default("nonstat_type").capitalize() + "NonStat"

    checkpoint_path = (
        f'ProximalPolicyOptimisation/Environments/'
        f'{parser.get_default("Variant")}-RND-{parser.get_default("RND")}'
        f'{parser.get_default("env_name")}{nonstat_suffix}/Weights'
    )

    parser.add_argument(
        '--checkpointDirectory', 
        type=str, 
        default=checkpoint_path,
        help='Where should the agent load and save the weights'
    )

    parser.add_argument('--AdvantageNormalisation', type=bool, default=False, help='Normalise the advantage estimates')
    parser.add_argument('--Network', type=str, default='Dense', help='Use Dense or CNN network (Dense for vector observations)')
    parser.add_argument('--Seed', type=int, default=42, help='Random seed')
    parser.add_argument('--Device', type=str, default=T.device('cuda' if T.cuda.is_available() else 'cpu'),
                        help='Device to run the neural networks (cpu or cuda)')
    
    parser.add_argument('--action_type', type=str, default='continuous', choices=['discrete', 'continuous'],
                        help='Type of action space: discrete or continuous')

    args = parser.parse_args()

    env, initial_obsShape = create_pyenv(env_name=args.env_name, Network=args.Network)


    env = apply_nonstationary_wrapper(env, {
    "nonstat": args.nonstat,
    "nonstat_type": args.nonstat_type,
    "var_target": args.var_target,
    "base_var": args.base_var,
    "delta_var": args.delta_var,
    "omegaVar": args.omegaVar,
    "interval": args.interval,
    "var_bodies": args.var_bodies
})

    env_name = args.env_name
    observation_space = env.observation_space
    action_space = env.action_space

    if args.action_type == 'discrete':
        n_actions = action_space.n 
    elif args.action_type == 'continuous':
        n_actions = action_space.shape[0]
    else:
        raise ValueError("Invalid action type. Expected 'discrete' or 'continuous'.")
    
    obs, _ = env.reset(seed=args.Seed)
    flat_obs = process_observation(obs)
    input_dims = flat_obs.shape

    Trainconfig = {
        "TrainingIterations": args.TrainingIterations,
        "UpdateFrequency": args.UpdateFrequency,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "policy_clip": args.policy_clip,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
    }

    # initialize_wandb(project_name="PPO-Project", run_name="TrainingRun", config=config)

    PPOAgent = Agent(
        variant=args.Variant,
        observation_space=observation_space,
        RND=args.RND,
        device=args.Device,
        action_space=action_space,
        n_actions=n_actions,
        input_dims=input_dims,
        NormaliseAdvantage=args.AdvantageNormalisation,
        Reward_Scaler=args.RewardScaler,
        chkpt_dir=args.checkpointDirectory,
        Network=args.Network,
        gamma=args.gamma,
        alpha=args.alpha,
        gae_lambda=args.gae_lambda,
        policy_clip=args.policy_clip,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        penaltyCoefficient=args.penaltyCoefficent,
        d_target=args.d_target,
        c1=args.c1,
        c2=args.c2,
        kl_div=args.KL_DIV,
        action_type=args.action_type
    )
    
    train_agent(agent=PPOAgent, env=env, seed=args.Seed, Iters=args.TrainingIterations,
                UpdateFrequency=args.UpdateFrequency, save_interval=args.save_interval,
                Record_Video=args.Record_Video, Rolling_Average=args.RollingAverage)
    #test_agent(agent=PPOAgent, env=env, Iters=args.TestingIterations, Rolling_Average=args.RollingAverage, seed=args.Seed)
