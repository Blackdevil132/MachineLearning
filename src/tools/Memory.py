import numpy as np

from collections import deque
from defines import *


class Memory:
    def __init__(self, max_size=MEMORY_SIZE):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def add_multiple(self, experiences):
        self.buffer += experiences

    def sample(self, batch_size=BATCH_SIZE):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)

        return [self.buffer[i] for i in index]
