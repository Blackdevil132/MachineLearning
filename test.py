from QRL import QRL
from GameEnemy import GameEnemy


env = GameEnemy()

done = 0
env.reset()
while not done:
    env.render()
    s, r, done, p = env.step(int(input("What do?")))
env.render()
