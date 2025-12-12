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
import pickle
from pathlib import Path

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

def initialize(raw_env: str, view=True):
    if view:
        out = p.connect(p.GUI)
    else:
        out = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    
    f_slider = p.addUserDebugParameter(paramName="Force of Limbs",rangeMin=0,rangeMax=200,startValue=10)
    kill_sim = p.addUserDebugParameter(paramName="Stop Simulation",rangeMin=1,rangeMax=-1, startValue=0) # If min > max, slider turns into a button.
    enable_fail_con = p.addUserDebugParameter(paramName="Enable Fail Condition",rangeMin=1,rangeMax=-1, startValue=0)

    return out, p.loadURDF(f"{raw_env}.urdf")

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

def reset_sim(view: bool, raw_env: str):
    p.disconnect()
    return initialize(view=view, raw_env=raw_env)

def save_data(data, name: str, dir: str):
    file_path=f"{dir}/{name}.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data(name: str, dir: str):
    file_path=Path(f"{dir}/{name}.pkl")
    if file_path.is_file():
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise AgentNotFound("Invalid Agent Name")

class Agent():
    def __init__(self, model, name, actions):
        self.name=name
        self.body=read_xacro(model)
        self.agent_id=None
        self.spawn()
        self.joints_dict={}
        self.joint_values=None
        self.labels=[]
        self.q_table=None
        self.actions=actions
        self.state_table=None
        self.num_joints= p.getNumJoints(self.agent_id)
        self.last_dist=0
        self.curr_dist=0
        self.initalize()
        #self.create_state_table()
        reward_function=get_dist_from_origin(self.agent_id)
    
    def get_name(self):
        return self.name
    
    def get_labels(self):
        return self.labels
        
    def get_id(self):
        return self.agent_id
    
    def get_joint_dict(self):
        return self.joints_dict
    
    def get_joint_values(self):
        return self.joint_values
    
    def feed_network(self):
        self.update_joint_values()
        return torch.tensor(self.joint_values).float()
    
    def create_q_table(self, num_of_actions: int):
        self.q_table = np.zeros(shape=(num_of_actions))
        
    def spawn(self):
        self.agent_id = p.loadURDF(f"{self.body}.urdf",startPos, startOrientation)
        p.changeDynamics(bodyUniqueId= self.agent_id, linkIndex=-1, lateralFriction=1)
    
    def initalize(self):
        for num in range (self.num_joints):
            if p.getJointInfo(self.agent_id, num)[2] != p.JOINT_FIXED:
                self.joints_dict[p.getJointInfo(self.agent_id, num)[0]]=(p.getJointInfo(self.agent_id, num)[1])
        self.labels=list(self.joints_dict.keys())
        self.update_joint_values()

    def update_joint_values(self):
        self.joint_values = np.array(list(map(lambda x: x[0], p.getJointStates(self.agent_id, jointIndices=self.labels))))
        return self.joint_values
    
    def move(self, direction, joint):
        dirs=[-1.2, 0, 1.2]
        p.setJointMotorControl2(bodyUniqueId = self.agent_id, 
                                jointIndex=joint, 
                                controlMode=p.POSITION_CONTROL, 
                                targetPosition=dirs[direction], 
                                force=max_force)
    
    def get_reward(self): # Lets make it so that it will give movement forward in units that value, and backwards the same
        new_dist=get_dist_from_origin(self.agent_id)
        reward = new_dist - self.last_dist
        self.last_dist=self.curr_dist
        self.curr_dist=new_dist
        return(reward)
    
    def reset(self):
        p.resetBasePositionAndOrientation(self.agent_id, posObj=startPos, ornObj=ornPos)
        

class AgentNotFound(Exception):
    """A custom exception class for specific errors."""
    pass

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


startPos = [0,0,0.25]
ornPos = [0,0,0.25, 0]
startOrientation = p.getQuaternionFromEuler([0,0,0])  

# Initial Params
current_agent=read_xacro("bodyv6")
current_environment ="plane"
physicsClient, planeId= initialize(current_environment, view=False)
time_interval = .1 #seconds
min_force, max_force = 1, 20
stats={"Dist": [], "Failed":[]}
actions={"forward": 1.2, "reset": 0, 'backward': -1.2}
duration=5 # Seconds
episodes=5

state_values = [-1.2, 0, 1.2]

actions = [-1.2, 0, 1.2] # dict(zip(['back', 'neutral', 'forward'], state_values))

replay_memory=[] #(State, Action, Reward, Next Action)

# Relevant sliders
f_slider = None
kill_sim = None
enable_fail_con = None


time_index_last = time.time()
prototype_agent = Agent(model='bodyv6', name='prototype', actions=actions)

states= 13

policy_1=NeuralNetwork(name='policy_1', num_actions=3, num_states=states)
target_1=NeuralNetwork(name='target_1', num_actions=3, num_states=states)

pol_optim = torch.optim.SGD(policy_1.parameters(), lr=0.01)
target_optim = torch.optim.SGD(target_1.parameters(), lr=0.01)


epsilon= .75 # this determines how 'curious' the agent will be
discount= 1 # this determines how important we deep future rewards [0-1]
update_steps= 4
joints = prototype_agent.get_labels()

actions={
        'forward': 1.2,
        'neutral': 0,
        'back': -1
        }

# save_data(data=prototype_agent, name='prototype_agent', dir='agents')
# prototype_nn.save('nn_models')

# Sampling: do actions
# Training: pick random replays

used=0
mse=0
replay_memory = ReplayMemory(max=10000)

s1=None
s2=None

for epi in range(0, 3):
    print(f"Episode: {epi}, Current Loss: {mse}")
    if epi == 39:
        see=True
    else:
        see=False
        
    physicsClient, planeId = reset_sim(view=see, raw_env=current_environment)
    prototype_agent.spawn()
    for t in range(0, 1000):
        choice=random.uniform(0, 1)
        joint=random.randint(0, 12)
        data=prototype_agent.feed_network()
        
        if epsilon >= choice:
            direction=random.sample([0, 1, 2], 1)[0]
        else:
            used+=1
            direction=policy_1.forward(data)[1]
        
        s1=prototype_agent.feed_network()

        prototype_agent.move(direction=direction, joint=joint)
        
        s2=prototype_agent.feed_network()
        
        replay_memory.push(s1, direction, prototype_agent.get_reward(), s2)
        
        p.stepSimulation()
        time.sleep(1./240.)
        
        # if p.readUserDebugParameter(kill_sim) > 0 or p.readUserDebugParameter(enable_fail_con) > 0:
        #     p.disconnect()
        #     break

    prototype_agent.reset()
        
    training_batch = replay_memory.sample(batch_size=60)
    sum=torch.zeros(size=(1, 3))
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
    
    if epi % 5 == 0:
        
        target_optim.step()  
    
print(mse)
        
        
        
            
    

