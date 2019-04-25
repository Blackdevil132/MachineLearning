import numpy as np
from src.Qtable import Qtable


# Qtable for 2-dim storing
class QtableEnemy(Qtable):
    def __init__(self, action_space, observation_space_1, observation_space_2):
        Qtable.__init__(self)
        self.action_space = action_space
        self.observation_space = (observation_space_1, observation_space_2)

        self.table = [{j: np.zeros(action_space) for j in range(observation_space_2)} for i in range(observation_space_1)]
        for i in range(self.observation_space[0]):
            self.table[i][255] = np.zeros(action_space)

    def get(self, state, action=None):
        if action is None:
            return self.table[state[0]][state[1]][:]

        return self.table[state[0]][state[1]][action]

    def update(self, state, action, newValue):
        self.table[state[0]][state[1]][action] = newValue

    def show(self):
        for dim1 in range(self.observation_space[0]):
            print("%i " % dim1, end='')
            for key in self.table[dim1].keys():
                print("\t%i: " % key, end='')
                for action in self.table[dim1][key]:
                    print("\t%.3f, " % action, end='')
                print()
