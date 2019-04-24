import numpy as np
from Qtable import Qtable


# Qtable for 2-dim storing
class QtableTime(Qtable):
    def __init__(self, action_space, observation_space_1, observation_space_2):
        Qtable.__init__(self)
        self.action_space = action_space
        self.observation_space = (observation_space_1, observation_space_2)

        self.table = [[np.zeros(action_space) for j in range(observation_space_2)] for i in range(observation_space_1)]

    def get(self, state, action=None):
        if action is None:
            return self.table[state[0]][state[1]][:]

        return self.table[state[0]][state[1]][action]

    def update(self, state, action, newValue):
        self.table[state[0]][state[1]][action] = newValue

    def show(self):
        for dim1 in range(self.observation_space[0]):
            print("%i " % dim1, end='')
            for dim2 in range(self.observation_space[1]):
                print("\t%i: " % dim2, end='')
                for action in self.table[dim1][dim2]:
                    print("\t%.3f, " % action, end='')
                print()
