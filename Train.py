"""
<Train.py>: Program will train model based on environment given.

The purpose of this program is to train a model using reinfocement learning. This is accomplished through the use of pybullet and pytorch.
PyTorch is used to create our neural networks, and PyBullet is a physics simulation that is specialized for reinforcement of robotics.

Basic DQN (Deep Q Nework) algorithm:
    1. An agent is loaded into some sort of simulation with 2 different neural networks: Target and Policy.
    2. In the simulation, an agent is observed over an episode (period of time) and evaluated after a training period (set number of episodes). The data collected per episode is state action reward action' (SARS).
    3. This is a batched training algorithm. A replay memory queue (FIFO) stores n SARS data and the training is as follows:
        a) Run a 
    4. We then move back to step 2 adjusting reward and hyperparameters while observing changes.

Decision making:
    In order to prevent model tunnelvision to a particular solution, a greedy epsilon algorithm is used. 
    This is like a regular greedy algo, prioritizing max reward, while maintaining epsilon randomness. Some random actions are taken in order to find potential innovation.

Author:   <Cj Pong/Hood Senior Project Team>
Created:  <02/03/2025>
Modified: <02/03/2025>
Version:  <0.0.0>
Contact:  pongcj@gmail.com

Dependencies:
    XacroDoc
    PyBullet
    PyTorch -> BulletClient extension
    Numpy
    Pandas

Usage:
    1. First create a sesion_#.json according to the sample (first 3 sections in JSON).
    2. Fill out parameters in main section.
    3. Run program to train data and observe results.
    4. Save data to retrain next session.
"""
import os
import pybullet as p
import time
import pybullet_data
from xacrodoc import XacroDoc
import torch
import random
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from pathlib import Path
from pybullet_utils import bullet_client as bc
import msgspec.json
import json
from pathlib import Path
from modules.data_processing import *
import torch.nn as nn
from collections import namedtuple, deque

class NeuralNetwork(nn.Module):
        def __init__(self, name, num_actions, num_states):
            super().__init__()
            self.name=name
            self.linear_relu_stack=nn.Sequential(
                nn.Linear(num_states, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
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
            a_t(float): Action chosen at step t.
            r_t(float): Calculated reward at step t.
            s_t+1(torch.tensor): Observed state at step t+1.
            
        '''
        self.memory.append((args))

    def sample(self, n):
        '''Returns sample of batch size n.'''
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)

class PhysClient():
    
    def __init__(self, name: str, render=True):
        self.name=name
        self.objects={}
        self.state_tensor=None
        start=time.time()
        mode=p.DIRECT
        print("\nLoading Client")
        if render: mode=p.GUI
        self.id = bc.BulletClient(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print(f"Finished! Time Elapsed: {time.time()-start}")
        self.set_gravity()
        
    def get_status(self):
        print(f"\nServer: {self.name} | Status: {self.id.isConnected()} | Entities: {self.objects.keys()}")
        
    def load_entities(self, file_names: list, agent_name: str, map='plane.urdf' ):
        self.id.loadURDF(map)
        for file in file_names:
            temp=self.id.loadURDF(read_xacro(file))
            self.objects[agent_name]=(temp, self.generate_joints_dict(temp))
            self.state_tensor=self.get_state(agent_name=agent_name)
            print(self.objects)
    
    def get_joint_dict(self, agent_name: str):
        return self.objects[agent_name][1]
    
    def set_gravity(self, grav=(0,0,-10)):
        self.id.setGravity(grav[0], grav[1], grav[2])
        print(f"\nServer {self.name} gravity set to: {grav}.")
    
    def get_state(self, agent_name: str) -> torch.tensor:
        # Joints are paired (pos, vel)
        return torch.tensor(np.array([(x[0], x[1]) for x in self.id.getJointStates(bodyUniqueId=self.objects[agent_name][0], jointIndices=list((self.objects[agent_name][1]).keys()))])).flatten()
    
    def get_pos(self, agent_name: str) -> tuple:
        return (self.id.getBasePositionAndOrientation(bodyUniqueId=self.objects[agent_name][0])[0])
    
    def move_joints(self, agent_name: str, joints: list, values: list, mode: int):
        move_type=(p.POSITION_CONTROL, p.VELOCITY_CONTROL, p.TORQUE_CONTROL)
        self.id.setJointMotorControlArray(self.objects[agent_name][0], joints, move_type[mode], values)
        
    def move_joint(self, agent_name: str, joint: int, value: float, mode: int):
        move_type=(p.POSITION_CONTROL, p.VELOCITY_CONTROL, p.TORQUE_CONTROL)
        self.id.setJointMotorControl2(self.objects[agent_name][0], joint, move_type[mode], value)
    
    def generate_joints_dict(self, id:int) -> dict:
        return {(self.id.getJointInfo(bodyUniqueId=id, jointIndex=joint))[0]:(self.id.getJointInfo(bodyUniqueId=id, jointIndex=joint))[1] for joint in range(self.id.getNumJoints(id)) if (self.id.getJointInfo(bodyUniqueId=id, jointIndex=joint))[2] != p.JOINT_FIXED }
    
    def generate_actions(self, values: tuple, agent_name: str):
        return [(x, y) for x in values for y in list((self.get_joint_dict(agent_name)).keys())]
    
    def step(self):
        self.id.stepSimulation()
        
    def clear(self):
        self.id.resetSimulation()
        self.id.loadURDF('plane.urdf')
        self.set_gravity()
        
    def disconnect(self):
        self.id.disconnect()
        print(f"\nServer {self.name} disconnected.")

class Simulation():
    
    def __init__(self, name: str,  num_of_clients: int, replay_mem: ReplayMemory, actions: dict, render=True):
        self.name=name
        self.gui_id=None
        self.rm=replay_mem
        self.clients={i: PhysClient(i, render=False) for i in range(num_of_clients)}
        if render: self.gui_id=PhysClient('GUI', render=True)
            
    def observe(self, file_names: list, episodes: int, server_id: PhysClient, agent_name: str, epsilon: float, action_values: tuple, render: bool, period=1000):
        server_id.load_entities(file_names, agent_name)
        actions=server_id.generate_actions(values=action_values, agent_name=agent_name)
        server_id.clear()
        
        for e in range(0, episodes):
        
            server_id.load_entities(file_names, agent_name)
            for i in range(period):
                print(server_id.objects)
                s1=server_id.get_state(agent_name=agent_name)
                if random.uniform(0, 1) > epsilon:
                    a=random.choice(actions)
                else:
                    a=random.choice(actions)
                print(a)
                server_id.move_joint(agent_name, joint=a[1], value=a[0], mode=0)
                for x in range(0, 10000): self.rm.push((s1, server_id.get_pos(agent_name), a, server_id.get_state(agent_name=agent_name)))
                server_id.step()
                if render: time.sleep(1./240.)
            
            server_id.clear()
            
    def learn():
        print("placeholder")
    
    
    def get_replay_mem(self):
        return self.rm
        
def initialize(dir_name: str, session_no: int) -> tuple:
    
    """Loads session data for a given scenario and session.

    Args:
        dir_name: Scenario data directory.
        session_no: Session Number.

    Returns:
        Hyperparameters - Dict \n
        Target Network Parameters - Dict \n
        Policy Network Parameters - Dict \n
    """
    data=load_json(dir_name, f'Session_{session_no}')
    return data["Hyperparameters"], data['Target'], data['Policy'] 


if __name__ == "__main__":
    
    actions={
        0: 1, #Move Forward 1 limb
        1: 1, #Move Backward 1 limb
        2: 1, #Dont Move 1 limb
    }
    
    dir_name='Basic_Walking'
    session_no=1
    network_dims=""
    memory=ReplayMemory(10000)
    
    hyp_params, policy, target = initialize(dir_name, session_no)
    session = Simulation("Test", render=True, num_of_clients=0, actions=actions, replay_mem=memory)
    session.observe(file_names=['simple_dude'], agent_name='dude', action_values=(-1.2, 0, 1.2), server_id=session.gui_id, epsilon=-1, render=True, period=1, episodes=1)
    print(session.get_replay_mem().memory)
    general_save(session.get_replay_mem(), 'Basic_Walking/replay_mems/')
    

