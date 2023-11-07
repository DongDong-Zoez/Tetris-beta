from board import BoardData
import gym
from gym import spaces
from gym.envs.registration import register
import cv2
from PIL import Image
import numpy as np
import random
import torch
from utils import pretty_print

class Tetris(gym.Env):

    def __init__(self, rows = 22, columns = 10, in_channels = 60, future_steps=4, future_shapes=5, verbose=0):
        super(Tetris, self).__init__()

        self.observation_space = spaces.Box(low=0, high=255, shape=(rows, columns, in_channels), dtype=np.uint8)
        self.action_space = spaces.Discrete(7)

        self.board = BoardData(rows, columns)
        self.action_mapping = {
            0: self.board.rotateRight,
            1: self.board.moveDown,
            2: self.board.moveLeft,
            3: self.board.moveRight,
            4: self.board.hold,
            5: self.board.rotateLeft,
            6: self.board.dropDown
        }
        self.action_name = ["Rotate Right", "Move Down", "Move Left", "Move Right", "Hold", "Rotate Left", "Drop Down"]

        self.action_stats = [0 for _ in range(7)]

        self.rows = rows 
        self.columns = columns
        self.in_channels = in_channels
        self.verbose = verbose

        self.lifetime = 0
        self.numHole = 0
        self.future_steps = future_steps
        self.future_shapes = future_shapes

        self.history_observation = [[0] * self.rows * self.columns for _ in range(future_steps)]
        self.currentDirection = -1
        self.currentShape = -1
        self.holdShape = -1
        self.nextShape = [[0 for _ in range(7)] for _ in range(5)]
        self.alreadyHold = False

        self.gameStatus = self.board.status()

        self.episodes = 0
        self.episodeReward = 0
        self.expectReward = 0

    def __to_dummy(self, index, length):
        dummy = [0 for _ in range(length)]
        dummy[index] = 1
        return dummy

    def reset(self):
        self.board.reset()
        self.board.createTetriminos()
        self.lifetime = 0
        self.numHole = 0
        return self.make_features(self.gameStatus)
    
    def make_features(self, observation):
        self.history_observation.pop(0)
        self.history_observation.append(observation["board"])
        self.currentDirection = self.__to_dummy(observation["currentDirection"], 4)
        self.currentShape = observation["currentShape"]
        self.holdShape = observation["holdShape"]
        self.alreadyHold = observation["alreadyHold"] * 1
        self.nextShape = observation["nextShape"]
        self.holeMask = observation["holeMask"]

        features = torch.ones((self.in_channels, self.rows, self.columns)) * 255
        index = 0
        for i in range(self.future_steps):
            features[index] = torch.tensor(self.history_observation[i]).view(self.rows, self.columns)
            index += 1
        for i in range(len(self.currentDirection)):
            features[index] *= self.currentDirection[i]
            index += 1
        for i in range(len(self.currentShape)):
            features[index] *= self.currentShape[i]
            index += 1
        for i in range(len(self.holdShape)):
            features[index] *= self.holdShape[i]
            index += 1
        for i in range(len(self.nextShape)):
            for j in range(self.future_shapes):
                features[index] *= self.nextShape[i][j]
                index += 1
        features[index] *= self.alreadyHold
        features[index + 1] *= (self.lifetime > 10 * 1)
        features[index + 2] *= self.holeMask

        return features.permute(1,2,0).numpy()
        # return features
    
    def calcReward(self, status, action):

        reward = 0
        lineElimination = status["lineElimination"]
        holeDeviation = status["holeDeviation"]
        alreadyHold = status["alreadyHold"]
        terminated = status["terminated"]
        holeMask = status["holeMask"]
        terminated = status["terminated"]
        height = status["height"]
        currentHeight = status["currentHeight"]

        self.action_stats[action] += 1

        # reward -= np.sum(np.abs(np.diff(height))) * 10
        reward -= 5 if action == 4 and alreadyHold else reward
        # reward = reward + 1 if action == 2 or action == 3 else reward
        reward = reward - 100 if terminated else reward
        # reward = reward - (np.max(height) - np.min(height)) if action == 6 else reward
        # reward = reward + min(currentHeight) / 2 if action == 6 else reward
        # reward -= np.max(height) / 10
        reward += 100 * (lineElimination ** 2)
        reward += holeDeviation
        # reward -= max(0, np.log(np.sum(holeMask) - self.numHole + 12)) * 2
        # self.numHole = np.sum(holeMask)

        return reward / 10
    
    def seed(self, seed):
        pass
    
    def render(self, mode="ai"):
        self.board.render()
        if mode == "human":
            lineElimination = self.gameStatus["lineElimination"]
            holeDeviation = self.gameStatus["holeDeviation"]
            alreadyHold = self.gameStatus["alreadyHold"]
            terminated = self.gameStatus["terminated"]
            print("Line eliminated: ", lineElimination, " " * 10)
            print("Hole deviation : ", holeDeviation, " " * 10)
            print("Num hole : ", self.numHole, " " * 10)
            print("Already hold   : ", alreadyHold, " " * 10)
            print("Terminated     : ", terminated, " " * 10)
            print("Episode        : ", self.episodes, "         ")
            print("Episode Reward : ", self.episodeReward, "         ")
            print("Except  Reward : ", self.expectReward, "         ")

    def close(self):
        pass

    def step(self, action):
        if action in self.action_mapping:
            self.action_mapping[action]()
        else:
            self.board.moveDown()
        self.lifetime += 1
        if action != 6:
            self.board.moveDown()
        else:
            self.lifetime = 0
        if self.lifetime > 20:
            action = 6
            self.board.dropDown()
            self.lifetime = 0
        self.gameStatus = self.board.status()
        reward = self.calcReward(self.gameStatus, action)
        self.episodeReward += reward
        if self.gameStatus["terminated"]:
            # self.expectReward = (self.expectReward * self.episodes + self.episodeReward) / (self.episodes + 1)
            self.expectReward = self.episodeReward
            self.episodeReward = 0
            self.episodes += 1
        actions = np.round(np.array(self.action_stats) / sum(self.action_stats),2)
        if self.verbose:
            print("Episode       : ", self.episodes, "         ")
            print("Episode Reward: ", self.episodeReward, "         ")
            print("Except  Reward: ", self.expectReward, "         ")
            # print("Action Stats  : \n", action_stats, "         ")
            print("Action Stats  : ", actions, "         ")
            print('\033[1;1H')
        return self.make_features(self.gameStatus), reward, self.gameStatus["terminated"], {}

register(
    id='Tetris-v0',
    entry_point='tetris:Tetris',
)