from GameEnemy import GameEnemy

game = GameEnemy(map_name="4x4")

game.reset()
game.render()

done = False
while not done:
    s, r, done, info = game.step(int(input("Where to go?")))
    game.render()
