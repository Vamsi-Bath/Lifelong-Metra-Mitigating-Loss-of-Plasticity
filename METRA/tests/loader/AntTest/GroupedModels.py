import os
import shutil
import tempfile
import time
import argparse

import numpy as np
import torch
import gymnasium as gym

from iod.RND.PPOAgent import Agent


MODEL_KEY = b"ant_loader_secret_key"


def xb(data: bytes, key: bytes) -> bytes:
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_models_dir():
    return os.path.join(get_script_dir(), "Models")


def get_backup_models_dir():
    return os.path.join(get_script_dir(), "BackupModels")


def resolve_algorithm_folder(algorithm: str) -> str:
    mapping = {
        "metra": "Metra",
        "lifelong_metra": "Lifelong_Metra",
        "dads": "DADS",
        "ppo": "PPO",
        "rnd": "RND",
    }
    return mapping.get(algorithm.lower(), algorithm)


def dct_tmp(source_dir: str, key: bytes):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Models directory not found: {source_dir}")

    temp_dir = tempfile.mkdtemp(prefix="temp_models_")

    for root, _, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        out_root = os.path.join(temp_dir, rel_path)
        os.makedirs(out_root, exist_ok=True)

        for filename in files:
            src_path = os.path.join(root, filename)

            if not os.path.isfile(src_path):
                continue

            with open(src_path, "rb") as f:
                e = f.read()

            d_data = xb(e, key)

            out_name = filename[:-4] if filename.endswith(".enc") else filename
            out_path = os.path.join(out_root, out_name)

            with open(out_path, "wb") as f:
                f.write(d_data)

    return temp_dir


def collect_models(search_dir):
    model_names = set()

    for root, _, files in os.walk(search_dir):
        for f in files:
            raw = f[:-4] if f.endswith(".enc") else f

            if raw.endswith("_actor.pth"):
                model_names.add(raw[:-len("_actor.pth")])

    return sorted(model_names)


def find_model_folder(base_dir, model_name):
    target_actor = f"{model_name}_actor.pth"

    for root, _, files in os.walk(base_dir):
        if target_actor in files:
            return root

    raise FileNotFoundError(f"Model folder not found for {model_name}")


def build_agent(env, device, models_dir, model_name):
    observation_space = env.observation_space
    action_space = env.action_space
    input_dims = observation_space.shape

    if hasattr(action_space, "n"):
        action_type = "discrete"
        n_actions = action_space.n
    else:
        action_type = "continuous"
        n_actions = action_space.shape[0]

    model_folder = find_model_folder(models_dir, model_name)

    print(f"[INFO] Observation Space: {observation_space}")
    print(f"[INFO] Action Space: {action_space}")
    print(f"[INFO] Input Dims: {input_dims}")
    print(f"[INFO] Num Actions: {n_actions}")
    print(f"[INFO] Action Type: {action_type}")
    print(f"[INFO] Model Name: {model_name}")
    print(f"[INFO] Model Folder: {model_folder}")

    agent = Agent(
        variant="PPO Clip",
        observation_space=observation_space,
        RND=False,
        device=device,
        action_space=action_space,
        n_actions=n_actions,
        input_dims=input_dims,
        Reward_Scaler=1,
        NormaliseAdvantage=False,
        chkpt_dir=model_folder,
        Network="Dense",
        gamma=0.99,
        alpha=3e-4,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        penaltyCoefficient=0.1,
        d_target=0.01,
        c1=0.5,
        c2=0.0,
        kl_div=0.0,
        action_type=action_type,
        model_name=model_name,
    )

    return agent


def run_test_episode_loop(agent, env, num_episodes=1, render_speed=0.02, verbose=True):
    episode_rewards = []

    # Fixed sampled skill for all episodes
    fixed_skill = np.random.uniform(-1.0, 1.0, size=(2,))
    print(f"[FIXED SKILL 2D] {fixed_skill}")

    for episode_idx in range(num_episodes):
        # New sampled context vector for each episode
        context_vector = np.random.uniform(-1.0, 1.0, size=(2,))
        print(f"[EPISODE {episode_idx + 1}] Context Vector 2D: {context_vector}")

        state, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, prob, value, entropy = agent.choose_action(
                state,
                deterministic=True,
            )

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step_count += 1

            if render_speed > 0:
                time.sleep(render_speed)

        episode_rewards.append(total_reward)

        if verbose:
            print(
                f"[EPISODE {episode_idx + 1}/{num_episodes}] "
                f"Reward: {total_reward:.2f} | Steps: {step_count}"
            )

    return episode_rewards


def test_models_from_directory(
    source_dir,
    model_name=None,
    num_episodes=1,
    render_speed=0.02,
    env_name="Ant-v5",
):
    print(f"[SOURCE DIR] {source_dir}")

    available_models = collect_models(source_dir)

    print(f"[MODELS FOUND] {available_models}")

    if not available_models:
        print("[ERROR] No models found.")
        return

    candidate_models = []

    if model_name and model_name in available_models:
        candidate_models.append(model_name)

    remaining = [m for m in available_models if m != model_name]
    np.random.shuffle(remaining)
    candidate_models.extend(remaining)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")

    temp_models_dir = None
    last_error = None

    try:
        temp_models_dir = dct_tmp(source_dir, MODEL_KEY)
        print(f"[TEMP MODELS DIR] {temp_models_dir}")

        for chosen_model in candidate_models:
            print(f"\n[TRY] {chosen_model}")

            env_instance = gym.make(env_name, render_mode="human")

            try:
                agent = build_agent(
                    env=env_instance,
                    device=device,
                    models_dir=temp_models_dir,
                    model_name=chosen_model,
                )

                agent.load_models()

                print(f"[OK] Loaded: {chosen_model}")

                rewards = run_test_episode_loop(
                    agent=agent,
                    env=env_instance,
                    num_episodes=num_episodes,
                    render_speed=render_speed,
                    verbose=True,
                )

                print("\n===== RESULTS =====")
                print(f"Model Used: {chosen_model}")
                print(f"Average Reward: {np.mean(rewards):.2f}")
                print(f"Std Reward: {np.std(rewards):.2f}")
                print(f"Min Reward: {np.min(rewards):.2f}")
                print(f"Max Reward: {np.max(rewards):.2f}")
                print("===================\n")

            except Exception as e:
                last_error = e
                print(f"[FAIL] {chosen_model}: {e}")

            finally:
                try:
                    env_instance.close()
                except Exception:
                    pass

        print("\n[DONE] Finished testing all models.")

        if last_error:
            print(f"[LAST ERROR] {last_error}")

    finally:
        if temp_models_dir and os.path.exists(temp_models_dir):
            shutil.rmtree(temp_models_dir, ignore_errors=True)


def testPretrain(
    algorithm,
    env,
    model_name=None,
    random_skill=None,
    cbp_mode="NoCBP",
    num_episodes=1,
):
    if env == "Humanoid":
        raise RuntimeError("Humanoid not supported")

    base_dir = get_models_dir()
    algorithm_folder = resolve_algorithm_folder(algorithm)
    algo_dir = os.path.join(base_dir, "Pretrain", algorithm_folder, cbp_mode)

    test_models_from_directory(
        source_dir=algo_dir,
        model_name=model_name,
        num_episodes=num_episodes,
    )


def testHierarhical(
    algorithm,
    env,
    model_name=None,
    random_skill=None,
    cbp_mode="NoCBP",
    num_episodes=1,
):
    if env == "Humanoid":
        raise RuntimeError("Humanoid not supported")

    base_dir = get_models_dir()
    algorithm_folder = resolve_algorithm_folder(algorithm)
    algo_dir = os.path.join(base_dir, "Hierarchical", algorithm_folder, cbp_mode)

    test_models_from_directory(
        source_dir=algo_dir,
        model_name=model_name,
        num_episodes=num_episodes,
    )


def testRND(env, model_name=None, num_episodes=1):
    if env == "Humanoid":
        raise RuntimeError("Humanoid is not supported.")

    source_dir = get_models_dir()

    available_models = collect_models(source_dir)
    available_models = [m for m in available_models if "_RND_" in m]

    print("[RND MODELS]", available_models)

    if not available_models:
        print("[ERROR] No RND models found.")
        return

    test_models_from_directory(
        source_dir=source_dir,
        model_name=model_name,
        num_episodes=num_episodes,
    )


def testBackupModels(model_name=None, num_episodes=1):
    backup_dir = get_backup_models_dir()

    test_models_from_directory(
        source_dir=backup_dir,
        model_name=model_name,
        num_episodes=num_episodes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["backup", "pretrain", "hierarchical", "rnd"],
        default="backup",
    )

    parser.add_argument("--algorithm", default="ppo")
    parser.add_argument("--env", default="Ant")
    parser.add_argument("--model", default=None)
    parser.add_argument("--cbp", default="NoCBP")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render_speed", type=float, default=0.02)

    args = parser.parse_args()

    if args.mode == "backup":
        testBackupModels(
            model_name=args.model,
            num_episodes=args.episodes,
        )

    elif args.mode == "pretrain":
        testPretrain(
            algorithm=args.algorithm,
            env=args.env,
            model_name=args.model,
            cbp_mode=args.cbp,
            num_episodes=args.episodes,
        )

    elif args.mode == "hierarchical":
        testHierarhical(
            algorithm=args.algorithm,
            env=args.env,
            model_name=args.model,
            cbp_mode=args.cbp,
            num_episodes=args.episodes,
        )

    elif args.mode == "rnd":
        testRND(
            env=args.env,
            model_name=args.model,
            num_episodes=args.episodes,
        )