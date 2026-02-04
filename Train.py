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
    1. First create a session_#.json according to the sample (first 3 sections in JSON).
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

Client_Ids = {}

class Simulation():
    def __init__(self, name: str):
        self.name=name
        self.a_ids={} #Agent Ids
        self.b_ids={} #Bullet Client Ids
    
    def initialize():
        print("placeholder")
    
    def get_status():
        print("placeholder")
    
    def get_phys_ids():
        print("placeholder")
        
    def placeholder():
        print("placeholder")
    
    
        
if __name__ == "__main__":
    
    dir_name='Basic_Walking'
    session_no=1
    network_dims=""
    
    hyp_params, policy, target = initialize(dir_name, session_no)
    
    
    
    
    

