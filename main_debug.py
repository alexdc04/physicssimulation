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
from data_processing import read_xacro, save_data, load_data, AgentNotFound
from pybullet_utils import bullet_client as bc

class Agent():
    def __init__(self, model:str, start_pos: list, phys_id):
        self.model=model
        self.start_pos=start_pos
        self.p=phys_id
        self.agent_id=self.p.loadURDF((read_xacro(model)), basePosition=self.start_pos)
        self.joints_dict={}
        self.generate_joint_info()
        self.state_tensor=None
        self.get_state()
        
    def generate_joint_info(self):
        if not self.joints_dict:
            for num in range (self.p.getNumJoints(self.agent_id)):
                if self.p.getJointInfo(self.agent_id, num)[2] != p.JOINT_FIXED:
                    self.joints_dict[self.p.getJointInfo(self.agent_id, num)[0]]=(self.p.getJointInfo(self.agent_id, num)[1]) 
        else:
            print("Already Generated")
                
    def get_joints(self) -> dict:
        return self.joints_dict
    
    def get_id(self) -> int:
        return self.agent_id
    
    def get_state(self) -> torch.Tensor:
        self.state_tensor = torch.tensor(list(map(lambda x: x[0], self.p.getJointStates(self.agent_id, jointIndices=list(self.joints_dict.keys())))))
        return self.state_tensor
    
    def check_pos(self) -> tuple:
        return self.p.getBasePositionAndOrientation(self.agent_id)
    
    def get_dist(self) -> float:
        xyz=self.p.getLinkState(self.agent_id, 0)[0]
        return math.sqrt((xyz[0])**2 + (xyz[1])**2)
    
    def spawn(self, phys_id):
        self.p.loadURDF(fileName=(read_xacro(self.model)), physicsClientId=phys_id, basePosition=self.start_pos)
        self.p.changeDynamics(bodyUniqueId= self.agent_id, linkIndex=-1, lateralFriction=1)
    
    def reset(self, phys_id):
        self.p.resetBasePositionAndOrientation(self.agent_id, posObj=startPos, ornObj=ornPos)
        
    def move(self, vel, joint):
        self.p.setJointMotorControl2(bodyUniqueId = self.agent_id, 
                                jointIndex=joint, 
                                controlMode=self.p.VELOCITY_CONTROL, 
                                targetVelocity=vel, 
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
        '''
        Save a transition (s, r, a, s).
        
        Args:
            s_t(torch.tensor): State at step t.
            r_t(float): Calculated reward at step t.
            a_t(float): Action chosen at step t.
            s_t+1(torch.tensor): Observed state at step t+1.
            
        '''
        self.memory.append((args))

    def sample(self, n):
        '''Returns sample of batch size n.'''
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)

class Environment(): 
    def __init__(self, map: str, render=False):
        self.render=render
        self.map_name=map
        self.p=self.start_sim()
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.map_id=self.p.loadURDF(fileName=f"{self.map_name}.urdf")
        self.set_grav()
        self.debug_features=[]
        
    def start_sim(self):
        '''Starts physics sim instance.'''
        if self.render:
            return bc.BulletClient(connection_mode=(p.GUI))
        else:
            return bc.BulletClient(connection_mode=(p.DIRECT))
        
    def get_p(self):
        '''Gives bullet client instance.'''
        return self.p
    
    def reload_map(self):
        '''Reloads map with its current map urdf.'''
        self.map_id=self.p.loadURDF(fileName=f"{self.map_name}.urdf")
    
    def set_grav(self, x=None, y=None, z=None):
        '''
        Sets gravity in physics instance.
        \nNote: This must be done when the sim is reloaded.
        
        Args:
            x(Float): X scalar for gravity vector.
            y(Float): Y scalar for gravity vector.
            z(Float): Z scalar for gravity vector.
        '''
        if x and y and z:
            self.p.setGravity(gravX=x, gravY=y, gravZ=z)
        else:
            self.p.setGravity(gravX=0, gravY=0, gravZ=-9.81)
            
    def stop(self):
        '''Stops physics sim instance.'''
        self.p.disconnect()
        
    def status(self) -> int:
        '''Checks status of given environment.'''
        if self.p.isConnected(): 
            print("Active Instance")
            return self.p.getNumBodies()
        else:
            print("Inactive Instance")
            return None
        
    def clear(self):
        '''Clears environment and reloads map.'''
        self.p.resetSimulation()
        self.reload_map()
        self.set_grav()
        
    def step(self):
        '''Steps the specific simulation. If rendered, will step at 1/240 a second.'''
        self.p.stepSimulation()
        if self.render: time.sleep(1./240.)

def dqn_train(vis_env: Environment, dir_env: Environment, agent: Agent, pn: NeuralNetwork, tn: NeuralNetwork, epsilon: float, 
                episodes: int, time_steps: int, view_interval: int, actions: list, batch_size: int):
    '''
    This method trains the DQN. It works in 2 stages:\n
    1. Choses action based on greedy epsilon algo. Saves each time step to Replay Memory.
    2. Trains the policy network based upon a random batch from Replay Memory. Target values chosen from target network.
    
    Args:
        vis_env(Environment): An environment to train and see visually.
        dir_env(Environment): An environment to simulate physics with no rendering.
        agent(Agent): Chosen agent.
        pn(NeuralNetwork): The main network that chooses the actions.
        tn(NeuralNetwork): Network used to train policy network.
        epsilon(float): Value 0-1 to determine 'curiosity'.
        episodes(int): Amount of episodes consisting of the sample and train stages.
        time_steps(int): Amount of steps the agent will observe in 1 episode.
        view_interval(int): The interval for using the vis_network.
        actions(list): List of actions possible.
        batch_size(int): How many samples to train on.
    '''
    
    
    
    print("placeholder")
    



startPos = [0,0,0.25]
ornPos = [0,0, 0.25, 0]
startOrientation = p.getQuaternionFromEuler([0,0,0])  
# Initial Params
current_environment ="plane"
time_interval = .1 #seconds
min_force, max_force = 1, 20
actions={"forward": 1.2, "reset": 0, 'backward': -1.2}



gui_sim=Environment(map=current_environment, render=True)
direct_sim=Environment(map=current_environment, render=False)

prototype_agent = Agent(model='bodyv7', start_pos=startPos, phys_id=gui_sim.get_p())

joints=list(prototype_agent.get_joints().keys())
states=len(joints)

policy_1=NeuralNetwork(name='policy_1', num_actions=4, num_states=states)
target_1=NeuralNetwork(name='target_1', num_actions=4, num_states=states)

replay_memory = ReplayMemory(max=10000)

pol_optim = torch.optim.SGD(policy_1.parameters(), lr=0.01)
target_optim = torch.optim.SGD(target_1.parameters(), lr=0.01)

epsilon= .75
discount= 1
update_steps= 4
episodes=100
batch_size=500
view_interval=5
time_steps=1000

actions=[-5, 0, 5]

def dqn_train(vis_env=gui_sim, dir_env=direct_sim, agent=prototype_agent, pn=policy_1, tn=target_1, epsilon=epsilon, 
                episodes=episodes, time_steps=time_steps, view_interval=view_interval, actions=actions, batch_size=batch_size):

























if __name__ == "__main__":
    
    while True:
        tick+=1
        prototype_agent.move(random.randint(-5, 5), random.sample(joints, 1)[0])
        gui_sim.step()
        
        if tick % 1000 == 0:
            print(f"Gui: {gui_sim.status()}\nDirect: {direct_sim.status()}")





























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
        
        
        
            
    

