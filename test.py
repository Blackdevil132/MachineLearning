from src.GameEnemy import GameEnemy
from src.QRL import QRL
from src.Game2Enemies import Game2Enemies

qrl = QRL(Game2Enemies(map_name="8x8"))
qrl.loadFromFile("qtable")
qrl.test_visual()

"""
done = 0
env.reset()
while not done:
    env.render()
    s, r, done, p = env.step(int(input("What do?")))
env.render()"""
