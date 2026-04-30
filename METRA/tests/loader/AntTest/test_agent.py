
import os
import shutil
import tempfile
import time
import numpy as np
import torch
import gymnasium as gym

from iod.RND.PPOAgent import Agent


MODEL_KEY = b"ant_loader_secret_key"


def xb(data: bytes, key: bytes) -> bytes:
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def get_e_models_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "Models")


def resolve_algorithm_folder(algorithm: str) -> str:
    mapping = {
        "metra": "Metra",
        "lifelong_metra": "Lifelong_Metra",
        "dads": "DADS",
        "ppo": "PPO",
        "rnd": "RND",
    }
    return mapping.get(algorithm.lower(), algorithm)


def get_model_names():
    models_dir = get_e_models_dir()

    if not os.path.exists(models_dir):
        return []

    model_names = set()

    for root, _, files in os.walk(models_dir):
        for f in files:
            raw_name = f[:-4] if f.endswith(".enc") else f
            if raw_name.endswith("_actor.pth"):
                model_names.add(raw_name[:-len("_actor.pth")])

    return sorted(model_names)


def d_models_to_temp(e_dir, key: bytes):
    if not os.path.exists(e_dir):
        raise FileNotFoundError(
            f"E models directory not found: {e_dir}"
        )

    temp_dir = tempfile.mkdtemp(prefix="temp_models_")

    for root, _, files in os.walk(e_dir):
        rel_path = os.path.relpath(root, e_dir)
        out_root = os.path.join(temp_dir, rel_path)
        os.makedirs(out_root, exist_ok=True)

        for filename in files:
            src_path = os.path.join(root, filename)

            if not os.path.isfile(src_path):
                continue

            with open(src_path, "rb") as f:
                e_data = f.read()

            d_data = xb(e_data, key)

            out_name = filename[:-4] if filename.endswith(".enc") else filename
            out_path = os.path.join(out_root, out_name)

            with open(out_path, "wb") as f:
                f.write(d_data)

    return temp_dir


def find_model_folder(base_dir, model_name):
    target_actor = f"{model_name}_actor.pth"

    for root, _, files in os.walk(base_dir):
        if target_actor in files:
            return root

    raise FileNotFoundError(f"Model folder not found for {model_name}")


def collect_models(search_dir):
    model_names = set()

    for root, _, files in os.walk(search_dir):
        for f in files:
            raw = f[:-4] if f.endswith(".enc") else f
            if raw.endswith("_actor.pth"):
                model_names.add(raw[:-len("_actor.pth")])

    return sorted(model_names)


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


def run_test_episode_loop(agent, env, num_episodes=10, render_speed=0.02, verbose=True):
    episode_rewards = []

    for episode_idx in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, prob, value, entropy = agent.choose_action(
                state, deterministic=True
            )

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            time.sleep(render_speed)

        episode_rewards.append(total_reward)

        if verbose:
            print(
                f"[EPISODE {episode_idx + 1}/{num_episodes}] "
                f"Reward: {total_reward:.2f} | Steps: {step_count}"
            )

    return episode_rewards


def testPretrain(algorithm, env, model_name=None, random_skill=None, cbp_mode="NoCBP"):
    if env == "Humanoid":
        raise RuntimeError("Humanoid not supported")

    base_dir = get_e_models_dir()
    algorithm_folder = resolve_algorithm_folder(algorithm)
    algo_dir = os.path.join(base_dir, "Pretrain", algorithm_folder, cbp_mode)

    if not os.path.exists(algo_dir):
        raise FileNotFoundError(
            f"No Pretrain folder for algorithm='{algorithm}' and cbp_mode='{cbp_mode}': {algo_dir}"
        )

    available_models = collect_models(algo_dir)

    print(f"[ALGO DIR] {algo_dir}")
    print(f"[MODELS FOUND] {available_models}")

    if not available_models:
        print("[ERROR] No Pretrain models found for this algorithm / CBP mode.")
        return

    candidate_models = []
    if model_name and model_name in available_models:
        candidate_models.append(model_name)

    remaining = [m for m in available_models if m != model_name]
    np.random.shuffle(remaining)
    candidate_models.extend(remaining)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")

    last_error = None
    models_dir = None

    try:
        models_dir = d_models_to_temp(base_dir, MODEL_KEY)

        for chosen_model in candidate_models:
            print(f"\n[TRY] {chosen_model}")
            env_instance = gym.make("Ant-v5", render_mode="human")

            try:
                agent = build_agent(env_instance, device, models_dir, chosen_model)
                agent.load_models()
                print(f"[OK] Loaded: {chosen_model}")

                run_test_episode_loop(
                    agent=agent,
                    env=env_instance,
                    num_episodes=100,
                    render_speed=0.02,
                    verbose=False,
                )

                print(f"[DONE] {chosen_model}")
                return

            except Exception as e:
                last_error = e
                print(f"[FAIL] {chosen_model}: {e}")

            finally:
                try:
                    env_instance.close()
                except Exception:
                    pass

        print("\n[FAIL] All Pretrain models failed.")
        if last_error:
            print("[LAST ERROR]", last_error)

    finally:
        if models_dir and os.path.exists(models_dir):
            shutil.rmtree(models_dir, ignore_errors=True)
            shutil.rmtree(models_dir, ignore_errors=True)


def testHierarhical(algorithm, env, model_name=None, random_skill=None, cbp_mode="NoCBP"):
    if env == "Humanoid":
        raise RuntimeError("Humanoid not supported")

    base_dir = get_e_models_dir()
    algorithm_folder = resolve_algorithm_folder(algorithm)
    algo_dir = os.path.join(base_dir, "Hierarchical", algorithm_folder, cbp_mode)

    if not os.path.exists(algo_dir):
        raise FileNotFoundError(
            f"No Hierarchical folder for algorithm='{algorithm}' and cbp_mode='{cbp_mode}': {algo_dir}"
        )

    available_models = collect_models(algo_dir)

    print(f"[ALGO DIR] {algo_dir}")
    print(f"[MODELS FOUND] {available_models}")

    if not available_models:
        print("[ERROR] No models found for this algorithm / CBP mode.")
        return

    candidate_models = []
    if model_name and model_name in available_models:
        candidate_models.append(model_name)

    remaining = [m for m in available_models if m != model_name]
    np.random.shuffle(remaining)
    candidate_models.extend(remaining)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")

    last_error = None
    models_dir = None

    try:
        models_dir = d_models_to_temp(base_dir, MODEL_KEY)

        for chosen_model in candidate_models:
            print(f"\n[TRY] {chosen_model}")
            env_instance = gym.make("Ant-v5", render_mode="human")

            try:
                agent = build_agent(env_instance, device, models_dir, chosen_model)
                agent.load_models()

                print(f"[OK] Loaded: {chosen_model}")

                rewards = run_test_episode_loop(
                    agent=agent,
                    env=env_instance,
                    num_episodes=1,
                    render_speed=0.02,
                )

                print("\n===== RESULTS =====")
                print(f"Model: {chosen_model}")
                print(f"CBP Mode: {cbp_mode}")
                print(f"Avg: {np.mean(rewards):.2f}")
                print(f"Std: {np.std(rewards):.2f}")
                print(f"Min: {np.min(rewards):.2f}")
                print(f"Max: {np.max(rewards):.2f}")
                return

            except Exception as e:
                last_error = e
                print(f"[FAIL] {chosen_model}: {e}")

            finally:
                try:
                    env_instance.close()
                except Exception:
                    pass

        print("\n[FAIL] All models failed.")
        if last_error:
            print("[LAST ERROR]", last_error)

    finally:
        if models_dir and os.path.exists(models_dir):
            shutil.rmtree(models_dir, ignore_errors=True)


def testRND(env, model_name=None):
    if env == "Humanoid":
        raise RuntimeError("Humanoid is not supported.")

    actual_model = model_name
    available_models = get_model_names()
    available_models = [m for m in available_models if "_RND_" in m]
    print("[RND MODELS]", available_models)

    if not available_models:
        print("[ERROR] No RND e models found in renameModels2.")
        return

    candidate_models = []
    if model_name in available_models:
        candidate_models.append(model_name)

    remaining_models = [m for m in available_models if m != model_name]
    np.random.shuffle(remaining_models)
    candidate_models.extend(remaining_models)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")

    e_models_dir = get_e_models_dir()
    print(f"[INFO] Candidate models: {candidate_models}")
    print(f"[INFO] E models dir: {e_models_dir}")

    last_error = None
    models_dir = None

    try:
        models_dir = d_models_to_temp(e_models_dir, MODEL_KEY)
        print(f"[INFO] Temporary d models dir: {models_dir}")

        for chosen_model in candidate_models:
            print(f"\n[TRY] Attempting model: {chosen_model}")
            env_instance = gym.make("Ant-v5", render_mode="human")

            try:
                agent = build_agent(env_instance, device, models_dir, chosen_model)
                agent.load_models()
                print(f"[OK] Loaded model: {chosen_model}")

                print(f"\n{'=' * 60}")
                print("[START] Testing for 100 episodes")
                print(f"Model: {chosen_model}")
                print(f"{'=' * 60}\n")

                episode_rewards = run_test_episode_loop(
                    agent=agent,
                    env=env_instance,
                    num_episodes=1,
                    render_speed=0.02,
                )

                print(f"\n{'=' * 60}")
                print("[RESULTS] Testing Complete")
                print(f"{'=' * 60}")
                print(f"Requested Model: {actual_model}")
                print(f"Model Used: {chosen_model}")
                print(f"Average Reward: {np.mean(episode_rewards):.2f}")
                print(f"Std Dev: {np.std(episode_rewards):.2f}")
                print(f"Min Reward: {np.min(episode_rewards):.2f}")
                print(f"Max Reward: {np.max(episode_rewards):.2f}")
                print(f"{'=' * 60}\n")
                return

            except FileNotFoundError as e:
                last_error = e
                print(f"[ERROR] Model not found: {chosen_model}")
                print(f"  {e}")

            except RuntimeError as e:
                last_error = e
                print(f"[RUNTIME ERROR] Model '{chosen_model}' failed:")
                print(f"  {e}")
                print("[INFO] Trying another model...")

            except Exception as e:
                last_error = e
                print(f"[ERROR] Model '{chosen_model}' failed:")
                print(f"  {e}")
                print("[INFO] Trying another model...")

            finally:
                try:
                    env_instance.close()
                except Exception:
                    pass

        print("\n[FAIL] All available RND models failed.")
        if last_error is not None:
            print(f"[LAST ERROR] {last_error}")

    finally:
        if models_dir and os.path.exists(models_dir):
            shutil.rmtree(models_dir, ignore_errors=True)