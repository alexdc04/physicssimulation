import pybullet as p
import time
import pybullet_data
from xacrodoc import XacroDoc
import os
import torch
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

def read_xacro(file_name: str) -> str:
    current_file = file_name

    doc = XacroDoc.from_file(f"models/xacro/{current_file}.urdf.xacro")

    urdf_string = doc.to_urdf_string()

    with open(f"models/raw/{current_file}.urdf", "w") as f:
            f.write(urdf_string)
            
    return "models/raw/" + file_name

def check_pos(agent: int) -> str:
    return p.getBasePositionAndOrientation(agent)

# Initial Params
current_robot=read_xacro("unicycle")
current_environment="plane"
time_interval = .1 #seconds

# Initialize the physics sim
physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
planeId = p.loadURDF(f"{current_environment}.urdf")
startPos = [0,0,1.25]
startOrientation = p.getQuaternionFromEuler([0,0,0])
agent = p.loadURDF(f"{current_robot}.urdf",startPos, startOrientation)
p.changeDynamics(bodyUniqueId= agent, linkIndex=-1, lateralFriction=1)

# Relevant sliders
f_slider = p.addUserDebugParameter(paramName="Force of Limbs",rangeMin=0,rangeMax=200,startValue=10)
kill_sim = p.addUserDebugParameter(paramName="Stop Simulation",rangeMin=1,rangeMax=-1, startValue=0) # If min > max, slider turns into a button.

# Gets active joints and indexes them with name in dict
numOfJoints = p.getNumJoints(agent)
joints_dict={}
for num in range (numOfJoints):
    if p.getJointInfo(agent, num)[2] != p.JOINT_FIXED:
        joints_dict[p.getJointInfo(agent, num)[0]]=(p.getJointInfo(agent, num)[1])
print(joints_dict)
joints=list(joints_dict.keys())



time_index_last = time.time()
while True:
    if time.time() - time_index_last >= time_interval:  
        p.setJointMotorControl2(agent, random.choice(joints), controlMode=p.POSITION_CONTROL, targetPosition=random.uniform(-1.2, 1.2), force=p.readUserDebugParameter(f_slider))
        time_index_last = time.time()
        
    p.stepSimulation()
    time.sleep(1./240.)
    
    if p.readUserDebugParameter(kill_sim) > 0:
        p.disconnect()
        break


