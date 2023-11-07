from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from tetris import Tetris
from model import QNet

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DQN Model")

    parser.add_argument("--device", type=str, default="cuda:6", help="Device for training")
    parser.add_argument("--timesteps", type=int, default=1e10, help="Number of timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Number of timesteps")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=600000, help="Replay buffer size")
    parser.add_argument("--learning_starts", type=int, default=0, help="Number of timesteps before learning starts")
    parser.add_argument("--target_update_interval", type=int, default=10000, help="Interval for target network updates")
    parser.add_argument("--train_freq", type=int, default=1000, help="Frequency of training steps")
    parser.add_argument("--save_freq", type=int, default=1e6, help="Frequency of saving model")
    parser.add_argument("--log_freq", type=int, default=10, help="Frequency of logging info")
    parser.add_argument("--exploration_final_eps", type=float, default=0.0, help="Final exploration epsilon")
    parser.add_argument("--exploration_fraction", type=float, default=1e-4, help="Exploration fraction")
    parser.add_argument("--features_dim", type=int, default=128, help="Dimension of features extracted by QNet")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--save_path", type=str, default="res2", help="Output dir to saving model")
    parser.add_argument("--tensorboard_log", type=str, default="./tensorboard/Tetris-v0/", help="Tensorboard log directory")

    args = parser.parse_args()
    return args

def main():
    # env_id = "Tetris-v0"
    # env = gym.make(env_id)
    env = Tetris()
    args = parse_args()

    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path=args.save_path)
    event_callback = EveryNTimesteps(n_steps=args.save_freq, callback=checkpoint_on_event)

    policy_kwargs = {
        "features_extractor_class": QNet,
        "features_extractor_kwargs": {
            "features_dim": args.features_dim
        }
    }

    model = DQN(
        "CnnPolicy", 
        env=env, 
        device=args.device,
        seed=args.seed,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        target_update_interval=args.target_update_interval,
        train_freq=args.train_freq,
        exploration_final_eps=args.exploration_final_eps,
        exploration_fraction=args.exploration_fraction,
        policy_kwargs=policy_kwargs,
        verbose=args.verbose,
        tensorboard_log=args.tensorboard_log
    )

    env.reset()
    model.learn(total_timesteps=args.timesteps, callback=event_callback, log_interval=args.log_freq)
    model.save(f'{args.save_path}/model')

    env.close()

if __name__ == "__main__":
    main()
