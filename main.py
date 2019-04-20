import QRL
import numpy as np
from tools import timeit


qrl = QRL.QRL(100000, 0.8, 0.95, 0.001)

timeit(qrl.run, [])
qrl.loadFromFile()

for elem in sorted(qrl.qtable.keys()):
    print(elem, qrl.qtable[elem])

qrl.environment.reset()
qrl.environment.render("human")

total_reward = 0
for i in range(100):
    steps = qrl.test(render=False)
    total_reward += np.sum(steps, axis=0)[3]

print("Average Reward over 100 Games: %.2f" % (total_reward/100.0))
