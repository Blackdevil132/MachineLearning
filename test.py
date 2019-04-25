from src.GameEnemy import GameEnemy
from src.QRL import QRL

qrl = QRL(GameEnemy(map_name="8x8"))
qrl.loadFromFile("qtables/190425_11")
qrl.test_visual()


"""
done = 0
env.reset()
while not done:
    env.render()
    s, r, done, p = env.step(int(input("What do?")))
env.render()"""
