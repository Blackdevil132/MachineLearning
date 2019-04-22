import QRL
import numpy as np
from tools import timeit
import sys

if len(sys.argv) < 5:
    #print("Usage: " + sys.argv[0] + " total_runs learning_rate discount_rate decay_rate")
    #exit(0)
    qrl = QRL.QRL(100000, 0.8, 0.9, 0.00001)
else:
    qrl = QRL.QRL(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))

timeit(qrl.run, [])
qrl.loadFromFile()

print("=== Q-Table ===============================")
for elem in sorted(qrl.qtable.keys()):
    print(elem[0], elem[1], qrl.qtable[elem])

print("====== MAP Layout ==================\n")
qrl.environment.reset()
qrl.environment.render("human")
print("\n")

total_reward = 0
for i in range(100):
    steps = qrl.test(render=False)
    for step in steps:
        total_reward += step[3]

print("===== Optimal Path =====================")
for i in range(len(steps)):
    print(i, steps[i])

print("\nAverage Reward over 100 Games: %.2f" % (total_reward/100.0))
print("Exploitation-Exploration Ratio: %i:%i\n" % (qrl.expexpratio[0], qrl.expexpratio[1]))

"""
print("===== Statistics Table ==============================")
for i in range(64):
    for j in range(20):
        print(i, j, qrl.statistics[bytes((i, j))])
"""