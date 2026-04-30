# Logger.py
import csv
import os

class CSVLoggerTrain:
    def __init__(self, logName):
        folder = "PPO_trainlog" 
        os.makedirs(folder, exist_ok=True)
        safe_logName = logName.replace("/", "_")
        self.file_path = os.path.join(folder, safe_logName + ".csv")


        self.file = open(self.file_path, mode='a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=[
            "episode", "timestep", "state", "action", "extrinsic_reward",
            "intrinsic_reward", "total_reward", "next_state",
            "actor_losses", "critic_losses", "entropy", "learn_step"
        ])
        self.writer.writeheader()
        

    def log(self, *,episode,timestep, state, action, extrinsic_reward, intrinsic_reward,total_reward, 
            next_state, total_loss=None, actor_losses=None, critic_losses=None, entropy=None, learn_step=None):
        self.writer.writerow({
            "episode": episode,
            "timestep": timestep,
            "state": state.tolist() if hasattr(state, "tolist") else state,
            "action": action,
            "extrinsic_reward": extrinsic_reward,
            "intrinsic_reward": intrinsic_reward,
            "total_reward": total_reward,
            "next_state": next_state,
            "actor_losses": actor_losses,
            "critic_losses": critic_losses,
            "entropy": entropy,
            "learn_step": learn_step
        })

class CSVLoggerTest:
    def __init__(self, logName):
        folder = "PPO_testlog" 
        os.makedirs(folder, exist_ok=True)


        safe_logName = logName.replace("/", "_")
        file_path = os.path.join(folder, safe_logName + ".csv")

        self.file = open(file_path, mode='a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=[
            "episode", "timestep", "state", "action", "extrinsic_reward",
            "intrinsic_reward", "total_reward", "next_state",
        ])
        self.writer.writeheader()
        

    def log(self, *,episode,timestep, state, action, extrinsic_reward, intrinsic_reward,total_reward, next_state):
            
        self.writer.writerow({
            "episode": episode,
            "timestep": timestep,
            "state": state.tolist() if hasattr(state, "tolist") else state,
            "action": action,
            "extrinsic_reward": extrinsic_reward,
            "intrinsic_reward": intrinsic_reward,
            "total_reward": total_reward,
            "next_state": next_state,
        })

