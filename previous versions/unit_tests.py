# import pybullet as p
# import time
# import pybullet_data
# from pybullet_utils import bullet_client as bc
from itertools import permutations, combinations, product
import numpy as np

# #test={"g1" : bc.BulletClient(p.GUI)} #or p.DIRECT for non-graphical version
# direct_client_ID_1 = bc.BulletClient(p.DIRECT)
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
# p.setGravity(0,0,-10)
# planeId = direct_client_ID_1.loadURDF("plane.urdf")
# startPos = [0,0,1]
# startOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = direct_client_ID_1.loadURDF("r2d2.urdf",startPos, startOrientation)

# for i in range (100000000):
#     p.stepSimulation()
#     time.sleep(1./240.)
    
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
# p.disconnect()
j_array=[0, 0, 0, 0, 0, 0]
joints=int(5)
vals=[j_array for x in range(joints)]
choices=[-1.2, 0, 1.2,]

states=({x: np.zeros(len(j_array)*len(choices)) for x in (list(product(choices, repeat=len(choices))))})

states[(-1.2, -1.2, 0)][4] = 4
for s in states.values():
    print(s)
print(len(states))