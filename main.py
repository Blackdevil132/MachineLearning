import QRL
import numpy as np
from tools import timeit
import sys
from GameEnemy import GameEnemy

IntToAction = ["LEFT", "DOWN", "RIGHT", "UP"]
mapname="4x4"

if len(sys.argv) < 5:
    # initialize standard parameters
    total_episodes = 200000
    learning_rate = 0.75
    discount_rate = 0.95
    decay_rate = 0.0001
else:
    # initialize with given parameter-values
    total_episodes = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    discount_rate = float(sys.argv[3])
    decay_rate = float(sys.argv[4])


qrl = QRL.QRL(env=GameEnemy(map_name=mapname), learning_rate=learning_rate, discount_rate=discount_rate, decay_rate=decay_rate)
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

numTests = 100
total_rewards = np.zeros(numTests)
for i in range(numTests):
    steps = qrl.test(render=False)
    for step in steps:
        total_rewards[i] += step[3]

    print("===== Test Game %i =====================" % (i+1))
    for j in range(len(steps)):
        step = steps[j]
        print(j, "Enemy at %i. Moving from %i %s to %i.\t Reward: %i." % (step[0][1], step[0][0], IntToAction[step[1]], step[2][0], step[3]))
    print("Total Reward: %i\n" % total_rewards[i])

print("\nAverage Reward over %i Games: %.2f" % (numTests, np.average(total_rewards)))
#print("Exploitation-Exploration Ratio: %i:%i\n" % (qrl.expexpratio[0], qrl.expexpratio[1]))

"""
print("===== Statistics Table ==============================")
for i in range(64):
    for j in range(20):
        print(i, j, qrl.statistics[bytes((i, j))])
"""