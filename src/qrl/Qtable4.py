import numpy as np

from src.qrl.Qtable import Qtable


# Qtable for 4-dim storing
class Qtable4(Qtable):
    def __init__(self, action_space, observation_space_1, observation_space_2, observation_space_3, observation_space_4):
        Qtable.__init__(self)
        self.action_space = action_space
        self.observation_space = (observation_space_1, observation_space_2, observation_space_3, observation_space_4)

        # TODO
        self.table = [{j: {k: np.zeros(action_space) for k in [e2 for e2 in range(observation_space_3)]} for j in [e1 for e1 in range(observation_space_2)]} for i in range(observation_space_1)]

    def get(self, state, action=None):
        if action is None:
            try:
                return self.table[state[0]][state[1]][state[2]][state[3]][:]
            except KeyError:
                print("Error at ", state[0], state[1], state[2], state[3])

        return self.table[state[0]][state[1]][state[2]][state[3]][action]

    def update(self, state, action, newValue):
        self.table[state[0]][state[1]][state[2]][state[3]][action] = newValue

    def show(self):
        for dim1 in range(self.observation_space[0]):
            print("%i " % dim1, end='')
            for dim2 in self.table[dim1].keys():
                print("%i " % dim2, end='')
                for dim3 in self.table[dim1][dim2].keys():
                    print("\t%i: " % dim3, end='')
                    for dim4 in self.table[dim1][dim2][dim3].keys():
                        print("\t%i, " % dim4, end='')
                        for action in self.table[dim1][dim2][dim3][dim4].keys():
                            print("\t%.3f, " % action, end='')
                    print()
