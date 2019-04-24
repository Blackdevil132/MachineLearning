from QRL import QRL

qrl = QRL(0,0,0,0)
qrl.loadFromFile("qtables/saved/190424_11")
print(qrl.qtable.get(bytes((6, 7))))

qrl.test(render=True)
