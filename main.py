import QRL
import numpy as np
from tools import timeit
import sys

IntToAction = ["LEFT", "DOWN", "RIGHT", "UP"]

if len(sys.argv) < 5:
    # initialize standard parameters
    total_episodes = 100000
    learning_rate = 0.8
    discount_rate = 0.95
    decay_rate = 0.0001
else:
    # initialize with given parameter-values
    total_episodes = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    discount_rate = float(sys.argv[3])
    decay_rate = float(sys.argv[4])


qrl = QRL.QRL(total_episodes, learning_rate, discount_rate, decay_rate)
exec_time = timeit(qrl.run, [total_episodes], 1)

# store exec_time in file
with open("performance.csv", 'a') as file:
    s = "%i,%.2f,%.2f,%.5f,%f;\n" % (total_episodes, learning_rate, discount_rate, decay_rate, exec_time)
    file.write(s)

print("=== Q-Table ===============================")
qrl.loadFromFile()
qrl.qtable.show()

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
    step = steps[i]
    print(i, "Moving from %i %s to %i.\t Reward: %i." % (step[0][0], IntToAction[step[1]], step[2][0], step[3]))

print("\nAverage Reward over 100 Games: %.2f" % (total_reward/100.0))
print("Exploitation-Exploration Ratio: %i:%i\n" % (qrl.expexpratio[0], qrl.expexpratio[1]))

"""
print("===== Statistics Table ==============================")
for i in range(64):
    for j in range(20):
        print(i, j, qrl.statistics[bytes((i, j))])
"""