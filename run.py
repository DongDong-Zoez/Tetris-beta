import argparse
from stable_baselines3 import DQN
import time

from tetris import Tetris

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Tetris RL model.")
    parser.add_argument('--model_path', type=str, default="res", help='Path to the RL model file')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model on')
    parser.add_argument('--sleep_time', type=float, default=0.003, help='Sleep time in seconds between rendering steps')
    parser.add_argument('--num_steps', type=int, default=37300000, help='Number of steps for the model')
    return parser.parse_args()

def main():
    args = parse_args()

    env = Tetris()
    obs = env.reset()

    model_path = f'{args.model_path}/rl_model_{args.num_steps}_steps'

    model = DQN.load(model_path, env=env, device=args.device)

    done = False
    step_count = 0
    while step_count < args.num_steps:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated = env.step(action)
        if done or truncated:
            obs = env.reset()
        env.render()
        time.sleep(args.sleep_time)
        step_count += 1
    env.close()

if __name__ == "__main__":
    main()
