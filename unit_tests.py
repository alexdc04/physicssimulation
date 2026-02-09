import pybullet as p
import time
import pybullet_data
from pybullet_utils import bullet_client as bc

#test={"g1" : bc.BulletClient(p.GUI)} #or p.DIRECT for non-graphical version
direct_client_ID_1 = bc.BulletClient(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = direct_client_ID_1.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = direct_client_ID_1.loadURDF("r2d2.urdf",startPos, startOrientation)

for i in range (100000000):
    p.stepSimulation()
    time.sleep(1./240.)
    
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()