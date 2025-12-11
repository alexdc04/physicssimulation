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
import torch.nn as nn
import torch.optim as optim

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

def initialize():
    out = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return out

def load_sim(raw_agent: str, raw_env: str) -> tuple:
# Initialize the physics sim

    p.setGravity(0,0,-9.81)
    print(f"{raw_env}.urdf here here here")
    startPos = [0,0,1.25]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    agentId = p.loadURDF(f"{raw_agent}.urdf",startPos, startOrientation)
    p.changeDynamics(bodyUniqueId= agentId, linkIndex=-1, lateralFriction=1)
    return agentId, p.loadURDF(f"{raw_env}.urdf"), time.time()

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

def episode(raw_agent: str, raw_env: str):
    p.resetSimulation()
    return load_sim(raw_agent, raw_env)

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )
    def forward(self, x):
        return self.fc(x)
    
# Initial Params
current_agent=read_xacro("bodyv6")
current_environment="plane"
physicsClient= initialize()
time_interval = .1 #seconds
min_force, max_force = 1, 200
stats={"Dist": [], "Failed":[]}
duration=5 # Seconds
episodes=5
agentId, planeId, start_time=load_sim(current_agent, current_environment)

# Relevant sliders
f_slider = p.addUserDebugParameter(paramName="Force of Limbs",rangeMin=0,rangeMax=200,startValue=10)
kill_sim = p.addUserDebugParameter(paramName="Stop Simulation",rangeMin=1,rangeMax=-1, startValue=0) # If min > max, slider turns into a button.
enable_fail_con = p.addUserDebugParameter(paramName="Enable Fail Condition",rangeMin=1,rangeMax=-1, startValue=0)

# Gets active joints and indexes them with name in dict
numOfJoints = p.getNumJoints(agentId)
joints_dict={}
for num in range (numOfJoints):
    if p.getJointInfo(agentId, num)[2] != p.JOINT_FIXED:
        joints_dict[p.getJointInfo(agentId, num)[0]]=(p.getJointInfo(agentId, num)[1])
# print(joints_dict)
joints=list(joints_dict.keys())
time_index_last = time.time()

# AI model will be able to choose how many joints, which joints, and how strong
# This current functions choices are just random
while True:
    # Failure Condition. If contact is made, sim exits.
    failure = p.getContactPoints(bodyA=planeId, bodyB=agentId, linkIndexA=-1, linkIndexB=-1 ) #base link = -1
    run_time = time.time() - start_time
    
    if time.time() - time_index_last >= time_interval:  
        
        time_index_last = time.time()
        move_multi_joints_random(agent=agentId, joint_indexes=joints)
        debug_stats(agent=agentId, show=False)
        
    p.stepSimulation()
    time.sleep(1./240.)
    
    if (run_time >= duration):
        agentId, planeId, start_time=episode(current_agent, current_environment)
    
    if p.readUserDebugParameter(kill_sim) > 0 or p.readUserDebugParameter(enable_fail_con) > 0:
        p.disconnect()
        break



for ep in range(episodes):
    failure = p.getContactPoints(bodyA=planeId, bodyB=agentId, linkIndexA=-1, linkIndexB=-1)
    runtime_start=time.time()
sum_stats()