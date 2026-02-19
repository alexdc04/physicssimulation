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

# imports
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

# neural network class
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
            x = self.linear_relu_stack(x.float())
            return x, torch.argmax(x).int()
        
        def save(self, dir: str):
            torch.save(self.state_dict(), f'{dir}/{self.name}.pth')
            
<<<<<<< HEAD
=======
# save a collection of states as a replay
>>>>>>> 7dbcdd167ba0f0420022909612699ba0386487e2
class ReplayMemory(object):
    
    # create a deque with a maximum length of max to store state information
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
        # append to memory
        self.memory.append((args))

    def sample(self, n):
        # returns sample of batch size n 
        return random.sample(self.memory, n)

    # returns length of memory deque
    def __len__(self):
        return len(self.memory)

# create physics client
class PhysClient():

    def __init__(self, name: str, render=True, sim_map = 'plane_urdf'):
        '''
        Initialize the physics client
        
        Variables:
            name: self-explanatory
            objects: a dictionary containing objects within the physics client
            state_tensor: initialized as Null
            start: start time
            mode: can be either DIRECT (w/o GUI) or GUI (starts w/ GUI, believe it or not)
            sim_map: what file to use for the base of the simulation
            id: refers to the PyBullet client
        '''
        self.name=name
        self.objects={}
        self.state_tensor=None
        start=time.time()
        mode=p.DIRECT
        print("\nLoading Client")
        if render: mode=p.GUI

        # create a new client
        self.id = bc.BulletClient(mode)

        # path for data files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print(f"Finished! Time Elapsed: {time.time()-start}")
        self.set_gravity()
        
    # check server name, current connected status, and what objects are present
    def get_status(self):
        print(f"\nServer: {self.name} | Status: {self.id.isConnected()} | Entities: {self.objects.keys()}")
        
    # load entities into the server from URDF/XACRO files
    def load_entities(self, file_names: list, agent_name: str):
        # load the map first
        self.id.loadURDF(self.sim_map)

        # for each file given in file_names, load the file, generate joints
        for file in file_names:
            temp=self.id.loadURDF(read_xacro(file))
            self.objects[agent_name]=(temp, self.generate_joints_dict(temp))
            self.state_tensor=self.get_state(agent_name=agent_name)
    
    # return joints belonging to a particular agent
    def get_joint_dict(self, agent_name: str):
        return self.objects[agent_name][1]
    
    # sets three-dimensional gravity for simulation (x, y, z)
    # a negative value will pull towards the origin from a given direction
    def set_gravity(self, grav=(0,0,-10)):
        self.id.setGravity(grav[0], grav[1], grav[2])
        #print(f"\nServer {self.name} gravity set to: {grav}.")
    
    # get the current state as a tensor
    def get_state(self, agent_name: str) -> torch.tensor:
        # Joints are paired (pos, vel)
        return torch.tensor(np.array([(x[0], x[1]) for x in self.id.getJointStates(bodyUniqueId=self.objects[agent_name][0], jointIndices=list((self.objects[agent_name][1]).keys()))])).flatten()
    
    # return position and orientation
    def get_pos(self, agent_name: str) -> tuple:
        return (self.id.getBasePositionAndOrientation(bodyUniqueId=self.objects[agent_name][0])[0])
    
    # to move multiple joints at once
    def move_joints(self, agent_name: str, joints: list, values: list, mode: int):
        move_type=(p.POSITION_CONTROL, p.VELOCITY_CONTROL, p.TORQUE_CONTROL)
        self.id.setJointMotorControlArray(self.objects[agent_name][0], joints, move_type[mode], values)
        
    # to move one joint at a time
    def move_joint(self, agent_name: str, joint: int, value: float, mode: int):
        move_type=(p.POSITION_CONTROL, p.VELOCITY_CONTROL, p.TORQUE_CONTROL)
        self.id.setJointMotorControl2(self.objects[agent_name][0], joint, move_type[mode], value)
    
    # create a dictionary of all joint information so long as joints are not fixed
    # joint id: information
    def generate_joints_dict(self, id:int) -> dict:
        return {(self.id.getJointInfo(bodyUniqueId=id, jointIndex=joint))[0]:(self.id.getJointInfo(bodyUniqueId=id, jointIndex=joint))[1] for joint in range(self.id.getNumJoints(id)) if (self.id.getJointInfo(bodyUniqueId=id, jointIndex=joint))[2] != p.JOINT_FIXED }
    
<<<<<<< HEAD
    def generate_actions(self, values: tuple, agent_name: str) -> list:
=======
    # what action will each joint take based on its joint_dict?
    def generate_actions(self, values: tuple, agent_name: str):
>>>>>>> 7dbcdd167ba0f0420022909612699ba0386487e2
        return [(x, y) for x in values for y in list((self.get_joint_dict(agent_name)).keys())]
    
    # move the simulation forward discretely
    def step(self):
        self.id.stepSimulation()
        
    # clear all objects and load map
    def clear(self):
        self.id.resetSimulation()
        self.id.loadURDF(self.sim_map)
        self.set_gravity()
        
    # disconnect server
    def disconnect(self):
        self.id.disconnect()
        print(f"\nServer {self.name} disconnected.")

class Simulation():
    
    def __init__(self, name: str,  num_of_clients: int, replay_mem: ReplayMemory, actions: dict, pol: NeuralNetwork, q: NeuralNetwork, render=True):
        self.name=name
        self.gui_id=None
        self.rm=replay_mem
        self.pol=pol
        self.q=q
        self.avg_dist=0
        self.clients={i: PhysClient(i, render=False) for i in range(num_of_clients)}
        if render: self.gui_id=PhysClient('GUI', render=True)
            
    def observe(self, file_names: list, episodes: int, server_id: PhysClient, agent_name: str, epsilon: float, action_values: tuple, render: bool, period=1000):
        server_id.load_entities(file_names, agent_name)
        actions=server_id.generate_actions(values=action_values, agent_name=agent_name)
        server_id.clear()
        
        for e in range(0, episodes):
            
            server_id.load_entities(file_names, agent_name)
            for i in range(period):
                
                s1=server_id.get_state(agent_name=agent_name)
                if random.uniform(0, 1) > epsilon:
                    a=random.choice(actions)
                else:
                    a=actions[int(self.q.forward(s1)[1])]
                server_id.move_joint(agent_name, joint=a[1], value=a[0], mode=0)
                self.rm.push((s1, server_id.get_pos(agent_name), a, server_id.get_state(agent_name=agent_name)))
                server_id.step()
                
                if render: time.sleep(1./240.)
            
            server_id.clear()
            
    def deep_Q_learn(self, batch_size: int, gamma: float):
        training_batch=self.rm.sample(batch_size)
        states=torch.stack([t[0][0] for t in training_batch])
        targets=torch.stack([t[0][3] for t in training_batch])
        n=len(training_batch)//2
        loss=nn.MSELoss()
        loss=(states, targets)
    
    def get_replay_mem(self):
        return self.rm
<<<<<<< HEAD

=======
        
>>>>>>> 7dbcdd167ba0f0420022909612699ba0386487e2
if __name__ == "__main__":
    
    device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
    )
    
    settings, situation, hyperparameters=initialize(dir_name='Basic_Walking', session_no='1')
    
    actions={
        0: 1,
        1: 1,
        2: 1
    }
    action_space=15
    observation_space=10
    
    dir_name='Basic_Walking'
    session_no=1
    network_dims=""
    memory=ReplayMemory(100000)
    policy_network=NeuralNetwork('pol', action_space, observation_space).to(device)
    q_network=NeuralNetwork('q', action_space, observation_space).to(device)
    q_network.load_state_dict(policy_network.state_dict())
    
    hyp_params, policy, target = initialize(dir_name, session_no)
    session = Simulation("Test", render=True, num_of_clients=1, actions=actions, replay_mem=memory, pol=policy_network, q=q_network)
    session.observe(file_names=['simple_dude'], agent_name='dude', action_values=(-1.2, 0, 1.2), 
                    server_id=session.gui_id, epsilon=.5, render=True, period=1000, episodes=2)
    session.learn(20, .6)
    
    general_save(session.get_replay_mem(), 'Basic_Walking/replay_mems/')
    

