import pybullet as p
import time
import pybullet_data
from xacrodoc import XacroDoc
import os
import torch
import math
import random
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import seaborn as sns

def read_xacro(file_name: str) -> str:
    current_file = file_name

    doc = XacroDoc.from_file(f"models/xacro/{current_file}.urdf.xacro")

    urdf_string = doc.to_urdf_string()

    with open(f"models/raw/{current_file}.urdf", "w") as f:
            f.write(urdf_string)
            
    return "models/raw/" + file_name

def check_pos(agent: int) -> str:
    return p.getBasePositionAndOrientation(agent)

def get_dist_from_origin(agent: int) -> float:
    xyz=p.getLinkState(agent, 0)[0]
    return math.sqrt((xyz[0])**2 + (xyz[1])**2)

# Initial Params
current_robot=read_xacro("dogv1")
current_environment="plane"
time_interval = .1 #seconds
min_force, max_force = 1, 200
start_time=time.time()
stats={"Dist": [], "Failed":[]}
duration=5 #Seconds

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
enable_fail_con = p.addUserDebugParameter(paramName="Enable Fail Condition",rangeMin=1,rangeMax=-1, startValue=0)

# Gets active joints and indexes them with name in dict
numOfJoints = p.getNumJoints(agent)
joints_dict={}
for num in range (numOfJoints):
    if p.getJointInfo(agent, num)[2] != p.JOINT_FIXED:
        joints_dict[p.getJointInfo(agent, num)[0]]=(p.getJointInfo(agent, num)[1])
# print(joints_dict)
joints=list(joints_dict.keys())
time_index_last = time.time()

# AI model will be able to choose how many joints, which joints, and how strong
# This current functions choices are just random
def move_multi_joints_random(agent: int, joint_indexes: list):
    k=random.randint(1, len(joint_indexes))
    moving_joint_indexes=random.sample(joint_indexes, k)
    joint_move_dists=[random.uniform(-1.2, 1.2) for n in moving_joint_indexes]
    joint_forces=[random.randint(min_force, max_force) for n in moving_joint_indexes]
    
    p.setJointMotorControlArray(agent, 
                                jointIndices=moving_joint_indexes, 
                                controlMode=p.POSITION_CONTROL, 
                                targetPositions=joint_move_dists, 
                                forces=joint_forces)

def debug_stats(agent: int, show=True):
    if len(failure) > 0:
        failed=True
    else:
        failed=False
    if show: print(f"Agent: {agent}\nFail Con. Status: {failed}\nFlat Dist From Origin: {get_dist_from_origin(agent)}\n")
    stats["Dist"].append(get_dist_from_origin(agent))
    stats["Failed"].append(failed)
    
def sum_stats():
    df=pd.DataFrame(stats)
    sum_graph=sns.relplot(
    data=df, kind="line",
    x=df.index, y=df["Dist"]
    )
    sum_graph.set_axis_labels("Ticks", "Meters") 
    plt.show()

while True:
    # Failure Condition. If contact is made, sim exits.
    failure = p.getContactPoints(bodyA=planeId, bodyB=agent, linkIndexA=-1, linkIndexB=-1 ) #base link = -1
    run_time = time.time() - start_time
    
    if time.time() - time_index_last >= time_interval:  
        
        time_index_last = time.time()
        move_multi_joints_random(agent=agent, joint_indexes=joints)
        debug_stats(agent=agent, show=False)
        
    p.stepSimulation()
    time.sleep(1./240.)
    
    if p.readUserDebugParameter(kill_sim) > 0 or p.readUserDebugParameter(enable_fail_con) > 0 or run_time >= duration:
        p.disconnect()
        break

sum_stats()

