from src.GameEnemy import GameEnemy


env = GameEnemy()

for p in env.P.keys():
    print(p, env.P[p])
"""
done = 0
env.reset()
while not done:
    env.render()
    s, r, done, p = env.step(int(input("What do?")))
env.render()"""
