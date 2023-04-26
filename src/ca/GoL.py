# Game of Life using CA

import pygame
from pygame.locals import *
from src.ca.CA import CA

class GoL(CA):
    def __init__(self, width, height, init_state):
        super().__init__(30, width, height, init_state)

    def get_neighborhood(self, i, j):
        neighborhood = 0
        for k in range(-1, 2):
            for l in range(-1, 2):
                neighborhood = neighborhood * 2 + self.state[(i + k) % self.width][(j + l) % self.height]
        return neighborhood

    def step(self):
        new_state = [[0 for _ in range(self.height)] for _ in range(self.width)]
        for i in range(self.width):
            for j in range(self.height):
                neighborhood = self.get_neighborhood(i, j)
                if neighborhood == 3 or (neighborhood == 4 and self.state[i][j] == 1):
                    new_state[i][j] = 1
        self.state = new_state
        self.states.append(new_state)

    def run(self, n):
        for _ in range(n):
            self.step()