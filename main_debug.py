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
import numpy as np
from pathlib import Path
from data_processing import read_xacro, save_data, load_data

class Agent():
    def __init__(self, model):
        self.model=model
        self.agent_id=p.loadURDF((read_xacro(model)))
        self.joints_dict={}
        self.generate_joint_info()
        self.state_tensor=None
        self.get_state()
        
    def generate_joint_info(self):
        if not self.joints_dict:
            for num in range (p.getNumJoints(self.agent_id)):
                if p.getJointInfo(self.agent_id, num)[2] != p.JOINT_FIXED:
                    self.joints_dict[p.getJointInfo(self.agent_id, num)[0]]=(p.getJointInfo(self.agent_id, num)[1]) 
        else:
            print("Already Generated")
                
    def get_joints(self) -> dict:
        return self.joints_dict
    
    def get_id(self) -> int:
        return self.agent_id
    
    def get_state(self) -> torch.Tensor:
        self.state_tensor = torch.tensor(list(map(lambda x: x[0], p.getJointStates(self.agent_id, jointIndices=list(self.joints_dict.keys())))))
        return self.state_tensor
    
    def check_pos(self) -> tuple:
        return p.getBasePositionAndOrientation(self.agent_id)
    
    def get_dist(self) -> float:
        xyz=p.getLinkState(self.agent_id, 0)[0]
        return math.sqrt((xyz[0])**2 + (xyz[1])**2)
    
    def spawn(self):
        p.loadURDF((read_xacro(self.model)))
    
    def reset(self):
        p.resetBasePositionAndOrientation(self.agent_id, posObj=startPos, ornObj=ornPos)
        
    def move(self, direction, joint):
        dirs=[-1.2, 0, 1.2]
        if not direction > 2:
            p.setJointMotorControl2(bodyUniqueId = self.agent_id, 
                                    jointIndex=joint, 
                                    controlMode=p.POSITION_CONTROL, 
                                    targetPosition=dirs[direction], 
                                    force=max_force)

class NeuralNetwork(nn.Module):
    def __init__(self, name, num_actions, num_states):
        super().__init__()
        self.name=name
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(num_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x, torch.argmax(x).int()
    
    def save(self, dir: str):
        torch.save(self.state_dict(), f'{dir}/{self.name}.pth')

class ReplayMemory(object):
    def __init__(self, max: int):
        self.memory = deque([], maxlen=max)
        
    def push(self, *args):
        """Save a transition"""
        self.memory.append((args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Simulation():
    def __init__(self, start_pos: list, orn_pos: list, min_force: int, max_force: int, environment: str, view=False):
        self.start_pos=start_pos
        self.orn_pos=orn_pos
        self.min_force=min_force
        self.max_force=max_force
        self.env=environment
        self.view=view
        self.plane_id=None
        self.physicsClient=None
        self.debug_features=[]
        self.start_sim()
        
    def start_sim(self):
        if self.view:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        self.spawn_env()
        
    def spawn_env(self):
        print(self.env)
        self.plane_id=p.loadURDF(f"{self.env}.urdf")
        
    def clear(self):
        p.resetSimulation()
        self.plane_id=p.loadURDF(f"{self.env}.urdf")
    
    def reset(self, view=False):
        p.disconnect()
        self.view=view
        self.start_sim()
    
    def stop():
        p.disconnect()

def dqn_train(epsilon: float, discount: float, episodes: int, update_step: int, batch_size: int, show_interval: int, 
                time_steps: int, pn: NeuralNetwork, tn: NeuralNetwork, po: torch.optim, tno: torch.optim, r_mem: ReplayMemory, 
                    sim: Simulation, agent: Agent):
    #initialize
    
    mse=0 # Mean Squared Error
    errors=[]
    joints=list(agent.get_joints().keys())
    actions=[0, 1, 2, 3]
    playback=False
    
    for episode in range(episodes):
        
        if episode % show_interval == 0:
            sim.reset(view=True)
            playback=True
        else:
            sim.clear()
            playback=False
        
        agent.spawn()
        
        for t in range(time_steps):
            
            joint=random.sample(joints, 1)[0]
            choice=random.uniform(0, 1)
            state=agent.get_state()
            
            if choice <= epsilon:
                action=random.sample(actions, 1)[0]
            else:
                action = pn.forward(state)[1]
                
            agent.move(direction=action, joint=joint)
            reward=agent.get_dist()
            replay_memory.push(state, action, reward, agent.get_state())
            p.stepSimulation()
            if playback: time.sleep(1./240.)
        
        training_batch = replay_memory.sample(batch_size=60)
        sum=torch.zeros(size=(1, 4))
        n=len(training_batch)//2
    
        for x in range(0, len(training_batch) , 2):
            step1=training_batch[x]
            step2=training_batch[x+1]
            
            guess=policy_1(step1[0])
            target=target_1(step2[0])
            
            sum+=(target[0] - guess[0])**2
            # Backward pass
        mse=(sum/n).sum()
        
        pol_optim.zero_grad() 
        mse.backward()  
        pol_optim.step()  
    
        if episode % 5 == 0:
            target_optim.step() 
            
        errors.append(mse)
    #train
    print(errors)
    sim.reset(view=True)
    while True:
        joint=random.sample(joints, 1)[0]
        choice=random.uniform(0, 1)
        state=agent.get_state()
        action = pn.forward(state)[1]
        agent.move(direction=action, joint=joint)
        p.stepSimulation()
        if playback: time.sleep(1./240.)

startPos = [0,0,0.25]
ornPos = [0,0,0.25, 0]
startOrientation = p.getQuaternionFromEuler([0,0,0])  
# Initial Params
current_environment ="plane"
time_interval = .1 #seconds
min_force, max_force = 1, 20
actions={"forward": 1.2, "reset": 0, 'backward': -1.2}

states= 13

current=Simulation(start_pos=startPos, orn_pos=ornPos, min_force=min_force, max_force=max_force, environment=current_environment, view=False)
prototype_agent = Agent(model='bodyv6')

print(prototype_agent.get_state())
policy_1=NeuralNetwork(name='policy_1', num_actions=4, num_states=states)
target_1=NeuralNetwork(name='target_1', num_actions=4, num_states=states)

replay_memory = ReplayMemory(max=10000)

pol_optim = torch.optim.SGD(policy_1.parameters(), lr=0.01)
target_optim = torch.optim.SGD(target_1.parameters(), lr=0.01)

epsilon= .75
discount= 1
update_steps= 4
episodes=10
batch_size=500
show_interval=100
time_steps=1000


dqn_train(epsilon=epsilon, discount=discount, episodes=episodes, update_step=update_steps, batch_size=batch_size, show_interval=show_interval, 
            time_steps=time_steps, pn=policy_1, tn=target_1, po=pol_optim, tno=target_optim, r_mem=replay_memory, sim=current, agent=prototype_agent)
































# # this determines how 'curious' the agent will be
#  # this determines how important we deep future rewards [0-1]
# 

# # Sampling: do actions
# # Training: pick random replays

# used=0
# mse=0


# s1=None
# s2=None

# # for epi in range(0, 3):
# #     print(f"Episode: {epi}, Current Loss: {mse}")
# #     if epi == 39:
# #         see=True
# #     else:
# #         see=False
        
# #     physicsClient, planeId = reset_sim(view=see, raw_env=current_environment)
# #     prototype_agent.spawn()
# #     for t in range(0, 1000):
# #         choice=random.uniform(0, 1)
# #         joint=random.randint(0, 12)
# #         data=prototype_agent.feed_network()
        
# #         if epsilon >= choice:
# #             direction=random.sample([0, 1, 2], 1)[0]
# #         else:
# #             used+=1
# #             direction=policy_1.forward(data)[1]
        
# #         s1=prototype_agent.feed_network()

# #         prototype_agent.move(direction=direction, joint=joint)
        
# #         s2=prototype_agent.feed_network()
        
# #         replay_memory.push(s1, direction, prototype_agent.get_reward(), s2)
        
# #         p.stepSimulation()
# #         time.sleep(1./240.)
        
# #         # if p.readUserDebugParameter(kill_sim) > 0 or p.readUserDebugParameter(enable_fail_con) > 0:
# #         #     p.disconnect()
# #         #     break

# #     prototype_agent.reset()
        
# #     training_batch = replay_memory.sample(batch_size=60)
# #     sum=torch.zeros(size=(1, 3))
# #     n=len(training_batch)//2
    
# #     for x in range(0, len(training_batch) , 2):
# #         step1=training_batch[x]
# #         step2=training_batch[x+1]
        
# #         guess=policy_1(step1[0])
# #         target=target_1(step2[0])
        
# #         sum+=(target[0] - guess[0])**2
# #         # Backward pass
# #     mse=(sum/n).sum()
    
# #     pol_optim.zero_grad() 
# #     mse.backward()  
# #     pol_optim.step()  
    
# #     if epi % 5 == 0:
        
# #         target_optim.step()  
    
# # print(mse)
        
        
        
            
    

