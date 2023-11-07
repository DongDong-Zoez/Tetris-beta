from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
import gym
import time
import torch
import torch.nn as nn

from tetris import Tetris
from model import QNet

from typing import Callable




def main():
    # env_id = "Tetris-v0"
    # env = gym.make(env_id)
    env = Tetris()
    # num_cpu = 8
    # env = SubprocVecEnv([lambda: env for _ in range(num_cpu)])

    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path="res/")
    event_callback = EveryNTimesteps(n_steps=1e5, callback=checkpoint_on_event)

    model = DQN(
        "CnnPolicy", 
        env=env, 
        device="cuda:7",
        learning_rate=5e-4,
        batch_size=2048,
        buffer_size=300000,
        learning_starts=0,
        target_update_interval=1000,
        train_freq=1000,
        policy_kwargs={
            "features_extractor_class": QNet,
            "features_extractor_kwargs": {
                "features_dim": 128
            },
            # "net_arch": [],
        },
        verbose=0,
        tensorboard_log="./tensorboard/Tetris-v0/"
    )
    # model.q_net.q_net = nn.Sequential(
    #     nn.Linear(7040, 7)
    # )
    # print(model.q_net)
    # return 

    obs = env.reset()
    model.learn(total_timesteps=1e10, callback=event_callback)
    model.save(f'res/model')

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    # print(f"Model {i}:", mean_reward, std_reward)
    env.close()

if __name__ == "__main__":
    main()
