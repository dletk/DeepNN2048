from BaseAI_3 import BaseAI
from math import *
import sys
import time

sys.setrecursionlimit(100000)

class PlayerAI(BaseAI):

    pre = 0.0
    currentMaxDepth = 1
    allowedTime = 0.2
    
    def getMove(self, grid):
        self.pre = time.clock()
        self.currentMaxDepth = 1
        move = None
        while time.clock() - self.pre < self.allowedTime:
            self.currentMaxDepth += 1
            newMove = self.decision(grid)
            if (move is None) or (newMove[1] > move[1] and newMove[0] is not None):
                move = newMove

        # print(time.clock() - self.pre)
        return move[0]

    def maximize(self, grid, alpha, beta, depth):
        availableMoves = grid.getAvailableMoves()
        depth = depth + 1

        if not availableMoves or depth > self.currentMaxDepth or time.clock() - self.pre > self.allowedTime:
            return None, self.heuristic(grid)

        maxChild = None
        maxScore = -sys.maxsize

        for direction in availableMoves:
            # Make a copy of the current grid
            tempGrid = grid.clone()
            tempGrid.move(direction)

            nextState = self.minimize(tempGrid, alpha, beta, depth)
            # nextState is a tuple with the form (direction, maxTile), for example: (1, 64)
            if nextState[1] > maxScore:
                maxScore = nextState[1]
                maxChild = direction
            if maxScore >= beta:
                break
            if maxScore > alpha:
                alpha = maxScore

        return maxChild, maxScore

    def minimize(self, grid, alpha, beta, depth):
        # Check to see if there is any available cell on the board
        emptyCells = grid.getAvailableCells()
        depth = depth + 1

        if not emptyCells or depth > self.currentMaxDepth or time.clock() - self.pre > self.allowedTime:
            return None, self.heuristic(grid)

        minChild = None
        minScore = sys.maxsize

        allStates = [(value, pos) for value in [2, 4] for pos in emptyCells]

        for state in allStates:
            tempGrid = grid.clone()
            tempGrid.insertTile(state[1], state[0])
            nextState = self.maximize(tempGrid, alpha, beta, depth)
            # next State is a tuple with the form (direction, maxTile), for example: (1, 64)
            if nextState[1] < minScore:
                minScore = nextState[1]
                minChild = tempGrid
            if minScore <= alpha:
                break
            if minScore < beta:
                beta = minScore

        return minChild, minScore

    def heuristic(self, grid):
        numEmptyCells = len(grid.getAvailableCells())
        sumCells = 0
        for row in grid.map:
            sumCells += sum(row)
        averageCellValue = sumCells/(16-numEmptyCells)
        monotonicity = self.monotonicity(grid)
        # print(monotonicity)

        emptyValue = log2(numEmptyCells) if numEmptyCells != 0 else 0
        largestPositionScore = self.largestTilePositionScore(grid)
        smoothness = self.smoothness(grid)

        return 2*monotonicity + emptyValue*2.5 + grid.getMaxTile() + 0.1*smoothness

    def monotonicity(self, grid):
        # Consider the monotonicity in all four direction
        monotonicity = [0, 0, 0, 0]    # up, down, left, right

        # Find monotonicity of left and right direction
        for row in grid.map:
            for index in range(3):
                currentCell = log2(row[index]) if row[index] != 0 else 0
                nextCell = log2(row[index+1]) if row[index+1] != 0 else 0
                # if nextCell > currentCell:
                monotonicity[2] += nextCell - currentCell
                # elif currentCell > nextCell:
                monotonicity[3] += currentCell - nextCell

        # Find monotonicity of up and down direction
        for row in range(3):
            for col in range(4):
                currentCell = log2(grid.map[row][col]) if grid.map[row][col] != 0 else 0
                nextCell = log2(grid.map[row+1][col]) if grid.map[row+1][col] != 0 else 0
                # if nextCell > currentCell:
                monotonicity[1] += nextCell - currentCell
                # elif currentCell > nextCell:
                monotonicity[0] += currentCell - nextCell

        return max(monotonicity[0], monotonicity[1]) + max(monotonicity[2], monotonicity[3])

    def largestTilePositionScore(self, grid):
        score = 0
        maxValue = grid.getMaxTile()
        for x in range(4):
            for y in range(4):
                if grid.getCellValue((x, y)) == maxValue:
                    if x in {0, 3} and y in {0, 3}:
                        score += 10
                    else:
                        score -= 10
        return score

    def smoothness(self, grid):
        """
        This function measure the absolute differences between neighboring tiles (only take into account the min), because
        the higher difference between tiles is, the worse is the tile, the return value need to be flip the sign
        For example: The total difference of the grid is 100, so its score is -100
        :param grid:
        :return: The score of difference between neighboring tiles (the higher the score, the better)
        """
        score = 0.0

        for row in range(4):
            for col in range(4):
                cellValue = grid.getCellValue((row, col))
                left = grid.getCellValue((row-1, col)) if row > 0 else 0
                right = grid.getCellValue((row+1, col)) if row < 3 else 0
                up = grid.getCellValue((row, col-1)) if col > 0 else 0
                down = grid.getCellValue((row, col+1)) if col < 3 else 0

                score += min([abs(cellValue-left), abs(cellValue - right), abs(cellValue - up), abs(cellValue - down)])

        return -score

    def smoothness2(self, grid):
        score = 0.0

        for row in range(4):
            for col in range(4):
                cellValue = grid.getCellValue((row, col))

                # Stackoverflow discussion
                rightMostPos = (row, col + 1)
                rightMostValue = 0
                while rightMostPos[1] <= 3:
                    if grid.getCellValue(rightMostPos) != 0:
                        rightMostValue = grid.getCellValue(rightMostPos)
                    rightMostPos = (rightMostPos[0], rightMostPos[1] + 1)

                downMostPos = (row + 1, col)
                downMostValue = 0
                while downMostPos[0] <= 3:
                    if grid.getCellValue(downMostPos) != 0:
                        downMostValue = grid.getCellValue(downMostPos)
                    downMostPos = (downMostPos[0] + 1, downMostPos[1])

                if cellValue != 0 and rightMostValue != 0:
                    score -= abs(log2(cellValue) - log2(rightMostValue))
                if cellValue != 0 and downMostValue != 0:
                    score -= abs(log2(cellValue) - log2(downMostValue))

    def findFarthestPosition(self, grid, cell, position, direction):
        previous = cell
        cell = grid.getCellValue((position[0]+direction[0], position[1] + direction[1]))

    def decision(self, grid):
        child = self.maximize(grid, -sys.maxsize, sys.maxsize, 0)
        return child
