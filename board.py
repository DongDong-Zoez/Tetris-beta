import random
import numpy as np
import itertools

class Tetriminos(object):

    shapeNone = 0
    shapeI = 1
    shapeJ = 2
    shapeL = 3
    shapeO = 4
    shapeS = 5
    shapeT = 6
    shapeZ = 7
    shapeAlpha = ["NN", "II", "JJ", "LL", "OO", "SS", "TT", "ZZ"]

    shapeStr = (
        "                ",
        "BbBbBbBbTtTtTtTt",
        "     JJ  JJJJJJ ",
        " LL      LLLLLL ",
        "  OOOO    OOOO  ",
        "   SSSS  SSSS   ",
        "   TT    TTTTTT ",
        " ZZZZ      ZZZZ "
    )

    shapeCoord = (
        #None
        (
            ((0, 0), (0, 0), (0, 0), (0, 0)), 
            ((0, 0), (0, 0), (0, 0), (0, 0)), 
            ((0, 0), (0, 0), (0, 0), (0, 0)), 
            ((0, 0), (0, 0), (0, 0), (0, 0)), 
        ),
        #I
        (
            ((0, 1), (1, 1), (2, 1), (3, 1)), 
            ((2, 0), (2, 1), (2, 2), (2, 3)), 
            ((0, 2), (1, 2), (2, 2), (3, 2)), 
            ((1, 0), (1, 1), (1, 2), (1, 3)), 
        ),
        #J
        (
            ((0, 0), (0, 1), (1, 1), (2, 1)), 
            ((1, 0), (2, 0), (1, 1), (1, 2)), 
            ((0, 1), (1, 1), (2, 1), (2, 2)), 
            ((1, 0), (1, 1), (0, 2), (1, 2)), 
        ),
        #L
        (
            ((2, 0), (0, 1), (1, 1), (2, 1)), 
            ((1, 0), (1, 1), (1, 2), (2, 2)), 
            ((0, 1), (1, 1), (2, 1), (0, 2)), 
            ((0, 0), (1, 0), (1, 1), (1, 2)), 
        ),
        #O
        (
            ((1, 0), (2, 0), (1, 1), (2, 1)), 
            ((1, 0), (2, 0), (1, 1), (2, 1)), 
            ((1, 0), (2, 0), (1, 1), (2, 1)), 
            ((1, 0), (2, 0), (1, 1), (2, 1)), 
        ),
        #S
        (
            ((1, 0), (2, 0), (0, 1), (1, 1)), 
            ((1, 0), (1, 1), (2, 1), (2, 2)), 
            ((1, 1), (2, 1), (0, 2), (1, 2)), 
            ((0, 0), (0, 1), (1, 1), (1, 2)), 
        ),
        #T
        (
            ((1, 0), (0, 1), (1, 1), (2, 1)), 
            ((1, 0), (1, 1), (2, 1), (1, 2)), 
            ((0, 1), (1, 1), (2, 1), (1, 2)), 
            ((1, 0), (0, 1), (1, 1), (1, 2)), 
        ),
        #Z
        (
            ((0, 0), (1, 0), (1, 1), (2, 1)), 
            ((2, 0), (1, 1), (2, 1), (1, 2)), 
            ((0, 1), (1, 1), (1, 2), (2, 2)), 
            ((1, 0), (0, 1), (1, 1), (0, 2)), 
        )
    )

    SRS = (
        # shapeNone
        (
            (
                ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)), # 0 --> 3
                ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)), # 0 --> 1
            ),
            (
                ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)), # 1 --> 0
                ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)), # 1 --> 2
            ),
            (
                ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)), # 2 --> 1
                ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)), # 2 --> 3
            ),
            (
                ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)), # 3 --> 2
                ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)), # 3 --> 0
            )
        ),
        # shapeI
        (
            (
                ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)), # 0 --> 3
                ((0, 0), (-2, 0), (1, 0), (-2, 1), (1, 2)), # 0 --> 1
            ),
            (
                ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)), # 1 --> 0
                ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)), # 1 --> 2
            ),
            (
                ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)), # 2 --> 1
                ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)), # 2 --> 3
            ),
            (
                ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)), # 3 --> 2
                ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)), # 3 --> 0
            )
        ),
        # shapeJLSTZ
        (
            (
                ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)), # 0 --> 3
                ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)), # 0 --> 1
            ),
            (
                ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)), # 1 --> 0
                ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)), # 1 --> 2
            ),
            (
                ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)), # 2 --> 1
                ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)), # 2 --> 3
            ),
            (
                ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)), # 3 --> 2
                ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)), # 3 --> 0
            ),
        )
    )

    shapeColor = [
        (0, 0, 0), # None
        (108, 234, 255), # I
        (51, 155, 255), # J
        (255, 186, 89), # L 
        (255, 255, 127), # O
        (132, 248, 128), # S
        (217, 88, 233), # T
        (255, 127, 121), # Z
    ]

class Shape(Tetriminos):

    def __init__(self, shape=0):
        self.shape = shape
        val = [0 for _ in range(7)]
        if self.shape != 0:
            val[self.shape - 1] = 1
        self.values = val

    def __str__(self):
        return Tetriminos.shapeStr[self.shape]
            
    def getCoords(self, direction, x, y):
        return ((x + dx, y + dy) for dx, dy in self.shapeCoord[self.shape][direction])
    
    def getBoundingOffsets(self, direction):
        currentCoords = Tetriminos.shapeCoord[self.shape][direction]
        minX, maxX, minY, maxY = 0, 0, 0, 0
        for x, y in currentCoords:
            if minX > x:
                minX = x
            if maxX < x:
                maxX = x
            if minY > y:
                minY = y
            if maxY < y:
                maxY = y
        return minX, maxX, minY, maxY
    
    def getRotationOffsets(self):
        if self.shape == Tetriminos.shapeNone or self.shape == Tetriminos.shapeO:
            return Tetriminos.SRS[0]
        elif self.shape == Tetriminos.shapeI:
            return Tetriminos.SRS[1]
        else:
            return Tetriminos.SRS[2]

class BoardData(object):

    def __init__(self, rows=22, columns=10):
        self.board = [0] * rows * columns
        self.rows = rows
        self.columns = columns
        self.currentX = -1
        self.currentY = -1
        self.currentDirection = 0
        self.permutationTable = list(itertools.permutations(range(1, 8)))
        self.permutation = [*random.choice(self.permutationTable)]
        self.nextPermutation = [*random.choice(self.permutationTable)]
        self.currentShape = Shape(0)
        self.nextShape = Shape(self.permutation.pop(0))
        self.permutation.append(self.nextPermutation.pop(0))
        self.holdShape = Shape(0)
        self.alreadyHold = False
        self.terminated = False

        self.lineElimination = 0
        self.holeDeviation = 0
        self.holeMask = 0
        self.preNumHole = 0
        self.height = [0 for _ in range(self.columns)]
        self.currentHeight = [0 for _ in range(4)]

    def createTetriminos(self):
        nextShape = Shape(self.permutation.pop(0))
        self.permutation.append(self.nextPermutation.pop(0))
        minX, maxX, minY, maxY = nextShape.getBoundingOffsets(0)
        # print(int(self.columns/2)-1, minY, maxY, np.array(self.board[:self.columns*2]).reshape(2, -1), self.nextShape.getCoords(0, 4, maxY))
        if self.sanityMove(nextShape, 0, int(self.columns/2)-1, 0):
            self.currentX = int(self.columns/2)-1
            self.currentY = minY
            self.currentDirection = 0
            self.currentShape = nextShape
            self.nextShape = Shape(self.permutation[0])
            self.nextPermutation = self.nextPermutation if self.nextPermutation else [*random.choice(self.permutationTable)]
            self.recordMove(self.currentShape, self.currentDirection, self.currentX, self.currentY)
            self.alreadyHold = False
        else:
            self.currentX = -1
            self.currentY = -1
            self.currentDirection = 0
            self.currentShape = Shape()
            self.terminated = True
    
    def move(self, currentShape, currentDirection, currentX, currentY, nextShape, nextDirection, nextX, nextY):
        self.removeHistory(currentShape, currentDirection, currentX, currentY)
        if self.sanityMove(nextShape, nextDirection, nextX, nextY):
            self.currentShape = nextShape
            self.currentDirection = nextDirection
            self.currentX = nextX
            self.currentY = nextY
            self.recordMove(nextShape, nextDirection, nextX, nextY)
            return True
        else:
            self.recordMove(currentShape, currentDirection, currentX, currentY)
            return False
        
    def hold(self):
        if self.holdShape.shape == 0:
            *_, minY, _ = self.nextShape.getBoundingOffsets(0)
            self.holdShape = self.currentShape
            if self.move(
                self.currentShape, 
                self.currentDirection, 
                self.currentX, 
                self.currentY, 
                self.nextShape, 
                0, 
                int(self.columns/2)-1, 
                -minY
            ):
                self.nextShape = self.permutation.pop(0)
                self.permutation.append(self.nextPermutation.pop(0))
                self.alreadyHold = True
            else:
                self.holdShape = Shape(0)
        elif not self.alreadyHold:
            *_, minY, _ = self.holdShape.getBoundingOffsets(0)
            temp = self.currentShape
            if self.move(
                self.currentShape, 
                self.currentDirection, 
                self.currentX, 
                self.currentY, 
                self.holdShape, 
                0, 
                int(self.columns/2)-1, 
                -minY
            ):
                self.alreadyHold = True
                self.holdShape = temp
        self.lineElimination = 0
        self.holeDeviation = 0
        
    def sanityMove(self, shape, direction, x, y):
        for dx, dy in shape.getCoords(direction, x, y):
            if dx >= self.columns or dx < 0 or dy >= self.rows or dy < 0:
                return False
            if self.board[dx + dy * self.columns] > 0:
                return False
        return True
    
    def dropDown(self):

        while self.move(
            self.currentShape,
            self.currentDirection, 
            self.currentX, 
            self.currentY, 
            self.currentShape,
            self.currentDirection,
            self.currentX, 
            self.currentY + 1
        ):
            pass
        loc = self.currentShape.getCoords(self.currentDirection, self.currentX, self.currentY)
        self.currentHeight = [y for _, y in loc]
        self.mergeTetriminos()
        self.removeFullLines()
        self.height = self.getColumnHeight(self.board)
        self.calcHoleDeviation(self.board)
        self.calcHoleMask(self.board)
        self.createTetriminos()
    
    def moveDown(self):
        nextY = self.currentY + 1
        self.move(
            self.currentShape,
            self.currentDirection, 
            self.currentX, 
            self.currentY, 
            self.currentShape,
            self.currentDirection, 
            self.currentX, 
            nextY
        )
        self.lineElimination = 0
        self.holeDeviation = 0
    
    def moveLeft(self):
        nextX = self.currentX - 1
        self.move(
            self.currentShape,
            self.currentDirection, 
            self.currentX, 
            self.currentY,
            self.currentShape, 
            self.currentDirection, 
            nextX, 
            self.currentY
        )
        self.lineElimination = 0
        self.holeDeviation = 0

    def moveRight(self):
        nextX = self.currentX + 1
        self.move(
            self.currentShape,
            self.currentDirection, 
            self.currentX, 
            self.currentY, 
            self.currentShape,
            self.currentDirection, 
            nextX, 
            self.currentY
        )
        self.lineElimination = 0
        self.holeDeviation = 0

    def rotateRight(self):
        nextDirection = (self.currentDirection + 1) % 4
        rotationOffsets = self.currentShape.getRotationOffsets()
        for dx, dy in rotationOffsets[self.currentDirection][1]:
            nextX = self.currentX + dx
            nextY = self.currentY - dy
            if self.move(
                self.currentShape,
                self.currentDirection, 
                self.currentX, 
                self.currentY,
                self.currentShape,
                nextDirection, 
                nextX, 
                nextY
            ):
                break
            else:
                continue
        self.lineElimination = 0
        self.holeDeviation = 0

    def rotateLeft(self):
        nextDirection = (self.currentDirection - 1) % 4
        rotationOffsets = self.currentShape.getRotationOffsets()
        for dx, dy in rotationOffsets[self.currentDirection][0]:
            nextX = self.currentX + dx
            nextY = self.currentY - dy
            if self.move(
                self.currentShape,
                self.currentDirection, 
                self.currentX, 
                self.currentY,
                self.currentShape,
                nextDirection, 
                nextX, 
                nextY
            ):
                break
            else:
                continue
        self.lineElimination = 0
        self.holeDeviation = 0

    def removeFullLines(self):
        newBoard = [0] * self.columns * self.rows
        newY = self.rows - 1
        lines = 0
        for y in range(self.rows - 1, -1, -1):
            rowsums = sum([1 if self.board[x + y * self.columns] > 0 else 0 for x in range(self.columns)])
            if rowsums < self.columns:
                for x in range(self.columns):
                    newBoard[x + newY * self.columns] = self.board[x + y * self.columns]
                newY -= 1
            else:
                lines += 1
        if lines > 0:
            self.board = newBoard
        self.lineElimination = lines
    
    def removeHistory(self, shape, direction, x, y):
        for dx, dy in shape.getCoords(direction, x, y):
            self.board[dx + dy * self.columns] = 0

    def eraseCurrent(self):
        self.removeHistory(self.currentShape, self.currentDirection, self.currentX, self.currentY)

    def getColumnHeight(self, board):
        mask = np.array(board) != 0
        heights = [0 for _ in range(self.columns)]
        for i in range(self.columns):
            board = mask[i::self.columns]
            for j in range(self.rows):
                if board[j]:
                    heights[i] = self.rows - j
                    break
        return heights
    
    def calcHoleMask(self, board):
        column_height = self.getColumnHeight(board)
        mask = np.zeros((self.rows, self.columns))
        for i, ch in enumerate(column_height):
            col = np.array(self.board[i::self.columns]) != 0
            temp = np.zeros(self.rows)
            temp[(self.rows-ch):] = 1
            mask[:,i] = temp - col
        self.holeMask = mask
    
    def calcHoleDeviation(self, board):
        column_height = self.getColumnHeight(board)
        max_column_height = np.max(column_height)
        mask = np.array(self.board[((self.rows - max_column_height)*self.columns):]) == 0
        self.holeDeviation = self.preNumHole - sum(mask)
        self.preNumHole = sum(mask)
    
    def recordMove(self, shape, direction, x, y):
        for dx, dy in shape.getCoords(direction, x, y):
            self.board[dx + dy * self.columns] = self.currentShape.shape

    def recordCurrent(self):
        self.recordMove(self.currentShape, self.currentDirection, self.currentX, self.currentY)

    def mergeTetriminos(self):
        self.recordMove(self.currentShape, self.currentDirection, self.currentX, self.currentY)

        self.currentX = -1
        self.currentY = -1
        self.currentDirection = 0
        self.currentShape = Shape()

    def reset(self):
        self.currentX = -1
        self.currentY = -1
        self.currentDirection = 0
        self.lineElimination = 0
        self.preNumHole = 0 
        self.holeDeviation = 0
        self.holeMask = 0
        self.height = [0 for _ in range(self.columns)]
        self.currentHeight = [0 for _ in range(4)]
        self.currentShape = Shape()
        self.terminated = False
        self.board = [0] * self.columns * self.rows
        self.permutation = [*random.choice(self.permutationTable)]
        self.nextPermutation = [*random.choice(self.permutationTable)]
        self.currentShape = Shape(0)
        self.nextShape = Shape(self.permutation.pop(0))
        self.permutation.append(self.nextPermutation.pop(0))

    def status(self):
        return {
            "board": self.board,
            "currentDirection": self.currentDirection,
            "currentShape": self.currentShape.values,
            "holdShape": self.holdShape.values,
            "alreadyHold": self.alreadyHold,
            "nextShape": [Shape(self.permutation[i]).values for i in range(7)],
            "lineElimination": self.lineElimination,
            "holeDeviation": self.holeDeviation,
            "holeMask": self.holeMask,
            "height": self.height,
            "currentHeight": self.currentHeight,
            "terminated": self.terminated,
        }

    def to_str(self):
        frame = ""
        lines = []
        for i in range(self.rows):
            lines.append("".join(map(lambda x: Tetriminos.shapeAlpha[x], self.board[self.columns*i:self.columns*(i+1)])))
        lines.append("XX" * self.columns)
        lines[0] = "XX" * 6 + lines[0]
        lines[1] = "XX" + str(self.holdShape)[:8] + "XX" + lines[1]
        lines[2] = "XX" + str(self.holdShape)[8:] + "XX" + lines[2]
        lines[3] = "XX" * 6 + lines[3]
        for i in range(4, len(lines)):
            lines[i] = "  " * 5 + "XX" + lines[i]
        for i in range(5):
            lines[i*3] += "XX" * 6
            lines[i*3+1] += "XX" + str(str(Shape(self.permutation[i]))[:8]) + "XX"     
            lines[i*3+2] += "XX" + str(str(Shape(self.permutation[i]))[8:]) + "XX"     
        lines[15] += "XX" * 6
        for i in range(16, len(lines)):
            lines[i] += "XX"
        frame = "\n".join(lines)
        return frame
    
    def render(self):
        frame = self.to_str()

        frame = frame.replace("XX", f'\033[48;2;128;128;128m  \033[0m')
        frame = frame.replace("NN", f'\033[48;2;0;0;0m  \033[0m')

        frame = frame.replace('Bb', f'\033[38;2;108;234;255m▄▄\033[0m')
        frame = frame.replace('Tt', f'\033[38;2;108;234;255m▀▀\033[0m')

        frame = frame.replace('ZZ', f'\033[48;2;255;127;121m  \033[0m')
        frame = frame.replace('LL', f'\033[48;2;255;186;89m  \033[0m')
        frame = frame.replace('OO', f'\033[48;2;255;255;127m  \033[0m')
        frame = frame.replace('SS', f'\033[48;2;132;248;128m  \033[0m')
        frame = frame.replace('II', f'\033[48;2;108;234;255m  \033[0m')
        frame = frame.replace('JJ', f'\033[48;2;51;155;255m  \033[0m')
        frame = frame.replace('TT', f'\033[48;2;217;88;233m  \033[0m')

        print('\033[1;1H', frame, sep='')

if __name__ == "__main__":
    board = BoardData()
    board.createTetriminos()
    board.dropDown()
    board.dropDown()