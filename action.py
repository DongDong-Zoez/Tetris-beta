import tty
import sys
import termios
from board import BoardData
import numpy as np
from tetris import Tetris

orig_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin)

# termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)    

class Action:
    none = 0
    moveLeft = 1
    moveRight = 2
    moveDown = 3
    dropDown = 4
    rotateLeft = 5
    rotateRight = 6
    hold = 7

env = Tetris()
obs = env.reset()
done = False

while True:

    if env.gameStatus["terminated"]:
        obs = env.reset()

    key = sys.stdin.read(1)[0]
    if key == "p":
        action = Action.rotateRight
        env.step(0)
    elif key == ";":
        action = Action.moveDown
        env.step(1)
    elif key == "l":
        action = Action.moveLeft
        env.step(2)
    elif key == "'":
        action = Action.moveRight
        env.step(3)
    elif key == "v":
        action = Action.hold
        env.step(4)
    elif key == "x":
        action = Action.rotateLeft
        env.step(5)
    elif key == " ":
        action = Action.dropDown
        env.step(6)
        # continue
    elif key == "r":
        break
    elif key == "q":
        # exit
        exit()
    else:
        action = Action.moveDown

    env.render(mode="human")

    # print('\033[1;1H', np.array(board.board).reshape(22,10), sep='')
    # print("Hole:", board.holeDeviation, "       ")

    # frame = board.to_str()

    # frame = frame.replace("XX", f'\033[48;2;128;128;128m  \033[0m')
    # frame = frame.replace("NN", f'\033[48;2;0;0;0m  \033[0m')

    # frame = frame.replace('Bb', f'\033[38;2;108;234;255m▄▄\033[0m')
    # frame = frame.replace('Tt', f'\033[38;2;108;234;255m▀▀\033[0m')

    # frame = frame.replace('ZZ', f'\033[48;2;255;127;121m  \033[0m')
    # frame = frame.replace('LL', f'\033[48;2;255;186;89m  \033[0m')
    # frame = frame.replace('OO', f'\033[48;2;255;255;127m  \033[0m')
    # frame = frame.replace('SS', f'\033[48;2;132;248;128m  \033[0m')
    # frame = frame.replace('II', f'\033[48;2;108;234;255m  \033[0m')
    # frame = frame.replace('JJ', f'\033[48;2;51;155;255m  \033[0m')
    # frame = frame.replace('TT', f'\033[48;2;217;88;233m  \033[0m')

    # # print(frame)

    # print('\033[1;1H', frame, sep='')