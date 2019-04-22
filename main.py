import QRL
import numpy as np
from tools import timeit
import sys

if len(sys.argv) < 5:
    #print("Usage: " + sys.argv[0] + " total_runs learning_rate discount_rate decay_rate")
    #exit(0)
    qrl = QRL.QRL(10000, 0.8, 0.9, 0.0002)
else:
    qrl = QRL.QRL(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))

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
print("Exploitation-Exploration Ratio: %i:%i" % (qrl.expexpratio[0], qrl.expexpratio[1]))


for i in range(len(qrl.statistics)):
    print(i, qrl.statistics[i])
