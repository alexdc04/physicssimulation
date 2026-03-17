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
from itertools import permutations, combinations, product
import math

# neural network class
class NeuralNetwork(nn.Module):
        def __init__(self, name, num_states, num_actions, ):
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

    def __init__(self, name: str, render=True, sim_map = 'plane.urdf'):
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
        self.sim_map=sim_map
        self.sim_map_id=None
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
        self.sim_map_id=self.id.loadURDF(self.sim_map)

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
    
    
    def get_state(self, agent_name: str) -> torch.tensor:
        return torch.tensor(np.array([(x[0], x[1]) for x in self.id.getJointStates(bodyUniqueId=self.objects[agent_name][0], jointIndices=list((self.objects[agent_name][1]).keys()))])).flatten()
    
    def get_state_tensor(self, agent_name: str) -> torch.tensor:
        # Joints are paired (pos, vel)
        return tuple(zip(*(self.id.getJointStates(bodyUniqueId=self.objects[agent_name][0], jointIndices=list((self.objects[agent_name][1]).keys())))))[:1]
    
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
    
    def generate_actions(self, values: tuple, agent_name: str) -> list:
        return [(x, y) for x in values for y in list((self.get_joint_dict(agent_name)).keys())]
    
    def check_collision(self, agent_name: str) -> tuple:
        return (self.id.getContactPoints(bodyA=self.objects[agent_name][0], bodyB=self.sim_map_id))
    
    def is_back_grounded(self, agent_name: str)-> bool:
        if any(c[3] == -1 for c in self.check_collision(agent_name=agent_name)): return True
        
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
    
    def get_vel(self, agent: str)-> tuple:
        return self.id.getBaseVelocity((self.objects[agent])[0])

class Simulation():
    
    def __init__(self, name: str,  num_of_clients: int, replay_mem: ReplayMemory, actions: dict, pol: NeuralNetwork, q: NeuralNetwork, render=True):
        self.name=name
        self.gui_id=None
        self.rm=replay_mem
        self.pol=pol
        self.pol_optim=torch.optim.SGD(self.pol.parameters())
        self.q=q
        self.q_optim=torch.optim.SGD(self.q.parameters())
        self.avg_dist=0
        self.clients={i: PhysClient(i, render=False) for i in range(num_of_clients)}
        self.training_cycles=0
        self.gui_id=PhysClient('GUI', render=render)
            
    def observe(self, file_names: list, episodes: int, server_id: PhysClient, agent_name: str, epsilon: float, action_values: tuple, render: bool, period=1000):
        server_id.load_entities(file_names, agent_name)
        actions=server_id.generate_actions(values=action_values, agent_name=agent_name)
        server_id.clear()
        for e in range(0, episodes):
            server_id.load_entities(file_names, agent_name)
            for i in range(period):
                
                s1=torch.tensor(server_id.get_state_tensor(agent_name=agent_name)[0])
                if random.uniform(0, 1) < epsilon:
                    a=random.choice(actions)
                else:
                    a=actions[(self.q(s1))[1]]
                    
                server_id.move_joint(agent_name, joint=a[1], value=a[0], mode=0)
                server_id.step()
                #to punish stillness
                if abs(server_id.get_vel(agent_name)[0][0]) > .2:
                    r=2**server_id.get_vel(agent_name)[0][0]
                else:
                    r=-1
                if server_id.is_back_grounded(agent_name=agent_name): r-=2
                self.rm.push((s1, a, sum(server_id.get_pos(agent_name))+r, torch.tensor(server_id.get_state_tensor(agent_name=agent_name)[0])))
                
                if render: time.sleep(1./240.)
            
            server_id.clear()
            
    def deep_Q_learn(self, batch_size: int, lr: float, loss: function):
        training_batch=self.rm.sample(batch_size)
        s1=self.q(torch.stack([t[0][0] for t in training_batch]))[0]
        s2=self.pol(torch.stack([t[0][3] for t in training_batch]))[0]
        yj=s2.clone()
        yj.scatter_reduce(dim=1, index=yj.argmax(dim=1).unsqueeze(1), src=torch.full((batch_size,1), lr), reduce="prod")
        yj.scatter_reduce_(dim=1, index=yj.argmax(dim=1).unsqueeze(1), src=torch.tensor([t[0][2] for t in training_batch]).unsqueeze(1), reduce="sum")
        n=len(training_batch)//2
        l=loss(s1, yj)
        l.backward()
        self.q_optim.step()
        self.q_optim.zero_grad()
        if self.training_cycles % 5 == 0 and self.training_cycles > 0:
            self.pol_optim.step()
            self.q_optim.zero_grad()
        self.training_cycles+=1
    
    def train(self, batch_size: int, lr: float, loss: function, file_names: list, episodes: int, server_id: PhysClient, agent_name: str, 
                epsilon: float, action_values: tuple, render: bool, eps_decay: float, iterations= 1, period=1000):
        start=time.time()
        
        for iteration in range(iterations):
            print(f"Starting iteration {iteration} of {iterations}. Total time elapsed: {time.time()-start}")
            temp=time.time()
            print(f"Starting Observation.")
            self.observe(file_names=file_names, agent_name=agent_name, action_values=action_values, server_id=server_id, epsilon=epsilon, render=render, period=period, episodes=episodes)
            print(f"Finished Observation! Time taken: {time.time()-temp}")
            temp=time.time()
            print(f"Starting Learning.")
            self.deep_Q_learn(batch_size=batch_size, lr=lr, loss=loss)
            print(f"Finished Learning! Time taken: {time.time()-temp}")
    
    def q_table_learn(self, values:tuple, actions, joints:int, epsilon: float):
        print("placeholder")
        #Create the 
        states={x: np.zeros(joints*actions) for x in list(product(values, repeat=joints))}
        x=random.uniform(0, 1)
        if x > epsilon:
            choice=divmod(random.randint(0, joints*actions))
        else:
            print("placeholder")
        
    def get_replay_mem(self):
        return self.rm

    def replay(self, file_names: list, server_id: PhysClient, agent_name: str, action_values: tuple, period: int, q):
        server_id.load_entities(file_names, agent_name)
        actions=server_id.generate_actions(values=action_values, agent_name=agent_name)
        server_id.clear()
        server_id.load_entities(file_names, agent_name)
        for x in range(period):
            a=actions[(q(torch.tensor(server_id.get_state_tensor(agent_name=agent_name)[0])))[1]]
            print(a)
            server_id.move_joint(agent_name, joint=a[1], value=a[0], mode=0)
            print((server_id.get_vel(agent=agent_name))[0][0], server_id.is_back_grounded(agent_name=agent_name))
            server_id.step()
            time.sleep(1./240.)
        
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
    observation_space=5
    render=True
    
    
    dir_name='Basic_Walking'
    session_no=1
    network_dims=""
    memory=ReplayMemory(100000)
    policy_network_path='Basic_Walking/model_params/pol'
    q_network_path='Basic_Walking/model_params/q'
    trainings=100
    rm_path=None #f'Basic_Walking\\mems\\116.pkl'
    epsilon=1
    train=False
    replay=True
    policy_network=NeuralNetwork('pol',  observation_space, action_space).to(device)
    q_network=NeuralNetwork('q', observation_space, action_space).to(device)
    q_network.load_state_dict(policy_network.state_dict())
    

    hyp_params, policy, target = initialize(dir_name, session_no)
    session = Simulation("Test", render=render, num_of_clients=1, actions=actions, replay_mem=memory, pol=policy_network, q=q_network)
    
    if train:
        if policy_network:
            policy_network.load_state_dict(torch.load(policy_network_path))
            q_network.load_state_dict(torch.load(q_network_path))
        if rm_path:
            memory=general_load(rm_path)
        
        for x in range(6): 
            epsilon-=.12
            session.train(file_names=['simple_dude'], agent_name='dude', action_values=(-1.2, 0, 1.2), server_id=session.gui_id, epsilon=1, eps_decay=0, render=render, 
                            period=500, episodes=10, batch_size=175, lr=.75, loss=nn.MSELoss(), iterations=30)
            trainings+=1
            
            rm_path=f'Basic_Walking\\mems\\1{trainings}.pkl'
            
            torch.save(policy_network.state_dict(), policy_network_path+str(trainings))
            torch.save(q_network.state_dict(), q_network_path+str(trainings))
            with open(rm_path, 'wb') as pkl_file:
                pickle.dump(memory, pkl_file)
                
    if replay:
        q=NeuralNetwork('', observation_space, action_space).to(device)
        for x in range(1, 6):
            r=16100+x
            q.load_state_dict(torch.load(q_network_path+str(r)))
            session.replay(file_names=['simple_dude'],server_id=session.gui_id, agent_name='dude', action_values=(-1.2, 0, 1.2), period=300, q=q)

