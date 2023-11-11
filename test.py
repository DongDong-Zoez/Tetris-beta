from tetris import Tetris

env = Tetris()
obs = env.reset()
env.step(6)
env.step(6)
env.step(6)
env.step(6)
obs, *_ = env.step(6)

print(obs.min())