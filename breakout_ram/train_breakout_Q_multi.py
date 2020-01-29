from multiprocessing import Process
import numpy as np
from trainer import RAM_Trainer


NUM_AGENTS = 4

agents = []

for i in range(NUM_AGENTS):
    a = RAM_Trainer()
    agents.append(a)


rewards = []
for a in agents:
    p = Process(target=a.train_episode())
    p.start()
    p.join()


for a in agents:
    print("-",a.rewards, a.done)

print("end")