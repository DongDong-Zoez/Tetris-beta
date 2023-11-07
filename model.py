import torch.nn as nn 
import torch.nn.functional as F

import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class QNet(BaseFeaturesExtractor):

    FILTERS = (60, 128, 256, 40)

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 7):
        super(QNet, self).__init__(observation_space, features_dim)

        in_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, QNet.FILTERS[0], 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(QNet.FILTERS[0], QNet.FILTERS[1], 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(QNet.FILTERS[1], QNet.FILTERS[2], 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(QNet.FILTERS[2], QNet.FILTERS[3], 1, 1, 0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            # nn.Identity()
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observation))
