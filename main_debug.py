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
from modules.data_processing import read_xacro, DriveConnection, load_session_data, save_session_data
from pybullet_utils import bullet_client as bc

class Agent():
    def __init__(self, model:str, start_pos: list, phys_id, vals: list):
        self.model=model
        self.start_pos=start_pos
        self.p=phys_id
        self.curr_x=0 
        self.last_x=0 
        self.traveled=0
        self.agent_id=self.p.loadURDF((read_xacro(model)), basePosition=self.start_pos)
        self.joints_dict={}
        self.generate_joint_info()
        self.state_tensor=None
        self.get_state()
        self.actions=define_actions(list(self.joints_dict.keys()), vals)
        
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
        self.state_tensor=(torch.tensor(list(map(lambda x: (x[0], x[1]), self.p.getJointStates(self.agent_id, jointIndices=list(self.joints_dict.keys()))))).flatten())
        return self.state_tensor
    
    def get_curr_pos(self) -> tuple:
        self.update_pos()
    
    def update_pos(self):
        self.last_x=self.curr_x
        self.curr_x=self.p.getLinkState(self.agent_id, 0)[0][0]
        self.traveled=self.last_x-self.curr_x
        
    def get_traveled(self):
        return self.traveled
    
    def get_dist(self) -> float:
        xyz=self.p.getLinkState(self.agent_id, 0)[0]
        return math.sqrt((xyz[0])**2 + (xyz[1])**2)
    
    def set_p(self, p):
        self.p=p
        
    def spawn(self):
        self.p.loadURDF(fileName=(read_xacro(self.model)), basePosition=self.start_pos)
        self.p.changeDynamics(bodyUniqueId= self.agent_id, linkIndex=-1, lateralFriction=1)
    
    def reset(self):
        self.p.resetBasePositionAndOrientation(self.agent_id, posObj=startPos, ornObj=ornPos)
        
    def get_actions(self):
        return self.actions
    
    def move(self, choice):
        a=self.actions[choice]
        self.p.setJointMotorControl2(bodyUniqueId = self.agent_id, 
                                jointIndex=a[1], 
                                controlMode=self.p.VELOCITY_CONTROL, 
                                targetVelocity=a[0], 
                                force=max_force)

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
    
    def get_render(self):
        return self.render
    
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
        
    def check_touching_ground(self, agent_id):
        out=1
        for x in range(15):
            if x == 14 or x == 11:
                pass
            else:
                if self.p.getContactPoints(bodyA=self.map_id, bodyB=agent_id, linkIndexA=-1, linkIndexB=x):
                    out=-1
        
        return out

def dqn_train(vis_env: Environment, dir_env: Environment, agent: Agent, pn: NeuralNetwork, tn: NeuralNetwork, epsilon: float, 
                episodes: int, time_steps: int, view_interval: int, actions: list, batch_size: int, replay: ReplayMemory, po: optim, to: optim):
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
    rewards=[]
    last_a=5
    decs=len(agent.get_actions())-1
    
    for episode in range(episodes):
        if episode % 10 == 0:
            epsilon -=.05
        mse=0
        curr_env=vis_env if episode % view_interval == 0 else dir_env
        curr_env.clear()
        
        agent.set_p(curr_env.get_p())
        agent.spawn()
        # Sample
        for t in range(time_steps):
            choice=random.uniform(0, 1)
            s1=agent.get_state()
            
            if epsilon >= choice:
                a=random.randint(0, decs)
            else:
                a=policy_1.forward(s1)[1]
                
            agent.move(a)
            curr_env.step()
            moved=agent.get_curr_pos()
            #r=reward(abs_dist=agent.get_dist(), x=moved[0], y=moved[1], z=moved[2], f=curr_env.check_touching_ground(agent.get_id()))
            r=(3*agent.get_dist())+(1.5*agent.get_traveled())
            rewards.append(r)
            last_a=a
            replay.push(s1, a, r, agent.get_state())
            
        curr_env.clear()
        
        training_batch = replay.sample(batch_size)
        sum=0
        n=len(training_batch)//2
        
        for x in range(0, len(training_batch) , 2):
            step1=training_batch[x]
            step2=training_batch[x+1]
            guess=pn(step1[0])[0][step1[1]]
            target=(tn(step2[0])[0]).max()
            
            sum+=((step2[2]+(discount*target)) - guess)**2
            # Backward pass
        mse=(sum/n)
        po.zero_grad() 
        mse.backward()  
        po.step()  
        
        if episode % 10 == 0:
            
            tn.load_state_dict(pn.state_dict())
        
        print(episode, mse)
    plt.plot(np.array(rewards))
    plt.show()
        
def reward(abs_dist: float):
    
    return ("temp")

def define_actions(joints: list, vals: list):
    act=[]
    index=0
    for v in vals:
        for j in joints:
            act.append((v,j))
    return act
startPos = [0,0,0.18]
ornPos = [0,0, 0.18, 0]
startOrientation = p.getQuaternionFromEuler([0,0,0])  
# Initial Params
current_environment ="plane"
time_interval = .1 #seconds
min_force, max_force = 1, 200
actions={"forward": 1.2, "reset": 0, 'backward': -1.2}
vals=[-15, -7, 7, 15]
# p_net_weights='nn_models/policy_1.pth'
# t_net_weights='nn_models/target_1.pth'

gui_sim=Environment(map=current_environment, render=True)
direct_sim=Environment(map=current_environment, render=False)
conn=DriveConnection()
prototype_agent = Agent(model='simple_dude', start_pos=startPos, phys_id=gui_sim.get_p(), vals=vals)

joints=list(prototype_agent.get_joints().keys())
states=len(joints)

act=define_actions(joints=joints, vals=vals)
index=0

policy_1=NeuralNetwork(name='policy_1', num_actions=len(act), num_states=states*2)
target_1=NeuralNetwork(name='target_1', num_actions=len(act), num_states=states*2)

replay_memory = ReplayMemory(max=10000)

pol_optim = torch.optim.SGD(policy_1.parameters(), lr=0.01)
target_optim = torch.optim.SGD(target_1.parameters(), lr=0.01)

data=load_session_data(hp_name='model1_HP.json', model_name='model1', conn=conn)

print(data)
# dqn_train(vis_env=gui_sim, dir_env=direct_sim, agent=prototype_agent, pn=policy_1, tn=target_1, epsilon=epsilon, 
#             episodes=episodes, time_steps=time_steps, view_interval=view_interval, actions=actions, batch_size=batch_size, replay=replay_memory,
#             po=pol_optim, to=target_optim)

data={  
        'model_name': 'first',
        'replay_memory': replay_memory,
        'policy_net_parameters': policy_1.state_dict,
        'target_net_parameters': target_1.state_dict
    }

save_session_data(data=data, conn=conn)
gui_sim.clear()
direct_sim.clear()

prototype_agent.set_p(gui_sim.get_p())
prototype_agent.spawn()
tick=0
print(prototype_agent.get_joints())


if __name__ == "__main__":
        
        tick+=1
        a=policy_1.forward(prototype_agent.get_state())[1]
        prototype_agent.get_curr_pos()
        print(prototype_agent.get_state())
        
        prototype_agent.move(a)
        gui_sim.step()
        
        
        if tick % 3000 == 0:
            print(prototype_agent.get_dist())
            gui_sim.clear()
            prototype_agent.spawn() 
