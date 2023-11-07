from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import gym
import time
import torch

from tetris import Tetris
from model import QNet

from typing import Callable

models_dir = 'res'
model_path = f'{models_dir}/rl_model_3850000_steps'

env = Tetris()
obs = env.reset()

model = DQN.load(model_path, env=env, device="cuda:1")

done = False
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated = env.step(action)
    if done or truncated:
        obs = env.reset()
    env.render()
    time.sleep(0.003)
env.close()