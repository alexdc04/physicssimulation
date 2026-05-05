from pybullet_utils import bullet_client as bc
import pybullet, time, pybullet_data, torch, random, bisect, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import itertools
from tqdm import tqdm
#can also connect using different modes, GUI, SHARED_MEMORY, TCP, UDP, SHARED_MEMORY_SERVER, GUI_SERVER
#pgui = bc.BulletClient(connection_mode=pybullet.GUI)

POSITION_CONTROL=0 
TORQUE_CONTROL=1
VELOCITY_CONTROL=2
device=torch.device('cuda')

class Replay_Buffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.index = 0
        self.size = 0

    def append(self, item):
        self.buffer[self.index] = item

        self.index = (self.index + 1) % self.capacity

        if self.size < self.capacity:
            self.size += 1
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def reset(self):
        self.size=0
        
class NeuralNetwork(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dims, policy):
        super().__init__()
        self.policy=policy
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(num_states, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims)
        )
        if policy:
            self.mean = nn.Linear(hidden_dims, num_actions)
            self.log_std = nn.Linear(hidden_dims, num_actions)
        else:
            self.value = nn.Linear(hidden_dims, 1)

    def forward(self, x):
        x=self.linear_relu_stack(x.float())
        if self.policy:
            mean=self.mean(x)
            log_std=torch.clamp(self.log_std(x), -1.5, 1.5)
            return (mean , log_std)
        else:
            return self.value(x)
        
    def save(self, dir: str, name):
        torch.save(self.state_dict(), f'{dir}/{name}.pth')

class Simulation():
    def __init__(self, num_of_envs, render, agent, longest=0):
        self.envs= {}
        self.pos=[0, 0, .21]
        self.connect(num_of_envs, render, agent)
        self.joints={self.envs[0].getJointInfo(1, joint)[1]:self.envs[0].getJointInfo(1, joint)[0] 
                        for joint in range(self.envs[0].getNumJoints(1)) 
                        if self.envs[0].getJointInfo(1, joint)[2] != 4}
        self.num_envs=num_of_envs
        if render: self.num_envs+=1
        self.longest_dist=longest
        
    def connect(self, num_of_envs, render, agent):
        for env in range(num_of_envs):
            p=bc.BulletClient(connection_mode=pybullet.DIRECT) 
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -10)
            p.loadURDF("plane.urdf")
            p.loadURDF(f'models\\raw\\{agent}.urdf', self.pos)
            self.envs[env] = p
            
        if render: 
            p=bc.BulletClient(connection_mode=pybullet.GUI) 
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -10)
            p.loadURDF("plane.urdf")
            p.loadURDF(f'models\\raw\\{agent}.urdf', self.pos)
            self.envs[-1] = p
    
    def reset_agent(self, env_idx):
        self.envs[env_idx].resetBasePositionAndOrientation(1, self.pos, [0, 0, 0, 1])
        self.envs[env_idx].resetBaseVelocity(1, [0, 0, 0], [0, 0, 0])
        for joint in self.joints.values():
            self.envs[env_idx].resetJointState(1, joint, 0)
    
    def move(self, forces, env_idx):
        self.envs[env_idx].setJointMotorControlArray(1, self.joints.values(), VELOCITY_CONTROL,  forces)
        return forces
    
    def get_collision(self, env_idx, joint_name):
        if self.envs[env_idx].getContactPoints(1, 0, self.joints[joint_name]):
            return True
        else:
            return False
    
    def get_state(self, env_idx):
        #Given an index, it will return 2 vectors, pos, vel
        return np.array([(i[0], i[1]) for i in self.envs[env_idx].getJointStates(1, self.joints.values())]).T
    
    def reward(self, env_idx, episode):
        x, y, z=self.envs[env_idx].getLinkState(1, self.joints[b'gaze'])[0]
        dist, h, p=self.envs[env_idx].getLinkState(1, self.joints[b'bend_low_torso'])[0]
        left_foot_touching=(self.get_collision(env_idx=env_idx, joint_name=b'left_ankle'))
        right_foot_touching=(self.get_collision(env_idx=env_idx, joint_name=b'right_ankle'))
        
        alive_bonus=1.45
        punish=-3
        
        if left_foot_touching:
            alive_bonus+=.5
            
        if right_foot_touching:
            alive_bonus+=.5
        
        if dist>self.longest_dist:
            print(f'New Record! {dist}')
            self.longest_dist=dist
            alive_bonus*=1.5
        
        if episode%50==100 and episode>0:
            punish*1.35
            alive_bonus*.90
            dist*=.86
        
        if z < .17:
            return punish, True
        else:
            return dist+alive_bonus, False
    
    def start(self, time_steps, episodes, discount, minibatch_len, lam, save_interval, save_dir, params):
        s0=self.get_state(0)
        rb=Replay_Buffer(capacity=time_steps)
        policy=NeuralNetwork(num_states=len(np.concatenate(s0)), num_actions=len(s0[0]), hidden_dims=128, policy=True).to(device)
        value=NeuralNetwork(num_states=len(np.concatenate(s0)), num_actions=1, hidden_dims=128, policy=False).to(device)
        
        if params:
            policy.load_state_dict(params['policy'])
            value.load_state_dict(params['value'])
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        value_optimizer  = torch.optim.Adam(value.parameters(),  lr=1e-3)
        
        for episode in range(episodes):
            alive={idx: [] for idx in self.envs.keys()}
            dones=np.stack([np.zeros(time_steps) for x in range(self.num_envs)])
            avg_reward=0
            for t in tqdm(range(time_steps)):
                s1, r_list = [], np.zeros(self.num_envs)

                for idx in self.envs.keys():
                    s1.append(torch.tensor(np.concatenate(self.get_state(env_idx=idx)), device=device))

                s1 = torch.stack(s1)

                with torch.no_grad():
                    mean, log_std = policy(s1)
                    dist = torch.distributions.Normal(mean, torch.exp(log_std))
                    a = dist.sample()
                    log_probs = dist.log_prob(a).sum(dim=-1)

                v_list = []

                for idx, act in zip(self.envs.keys(), a):
                    self.move(act, idx)
                    self.envs[idx].stepSimulation()

                    r, fail= self.reward(idx, episode)
                    r_list[idx] = r

                    if fail:
                        self.reset_agent(idx)

                with torch.no_grad():
                    for idx in self.envs.keys():
                        v_list.append(value(torch.tensor(np.concatenate(self.get_state(idx)),device=device)))

                v_stack = torch.stack(v_list)
                avg_reward+=sum(r_list)
                rb.append([s1.detach(), a.detach(), log_probs.detach(), torch.tensor(r_list, device=device).detach(), v_stack.detach()])
                
            gae = 0
            rb.buffer[time_steps-1].append(torch.tensor(gae).float().detach())
            for t in range(time_steps-2, -1, -1):
                delta = (rb.buffer[t][3]+ discount * rb.buffer[t + 1][4].squeeze(-1)- rb.buffer[t][4].squeeze(-1))
                gae = delta + discount * lam * gae
                rb.buffer[t].append(gae.detach())
                
            # returns
            for t in range(time_steps):
                rb.buffer[t].append((rb.buffer[t][5] + rb.buffer[t][4].squeeze(-1)).detach())

            sample = rb.sample(minibatch_len)

            states=torch.stack([x[0] for x in sample]).to(device).detach()
            actions=torch.stack([x[1] for x in sample]).to(device).detach()
            old_logprob=torch.stack([x[2] for x in sample]).to(device).detach()
            advantages=torch.stack([x[5] for x in sample]).to(device).detach()
            returns=torch.stack([x[6] for x in sample]).to(device).detach()

            mean, log_std = policy(states)
            dist = torch.distributions.Normal(mean, torch.exp(log_std))
            log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(log_probs - old_logprob)
            eps = 0.2
            policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - eps, 1 + eps) * advantages).mean()
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            value_pred = value(states).squeeze(-1)
            value_loss = F.mse_loss(value_pred, returns.float())
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()    
            
            
            if episode%save_interval==0:
                save={'policy':policy.state_dict(), 'value':value.state_dict()}
                with open(f'{save_dir}\\weights{episode}.pkl', 'wb') as file:
                    pickle.dump(save, file)
                
            print(f'Avg-Reward:{avg_reward/self.num_envs}\nValue Loss:{value_loss}\nPolicy Loss:{policy_loss}\nLongest Dist:{self.longest_dist}')        
            
if __name__=='__main__':
    with open('PPO Trial 1\\weights20.pkl', 'rb') as file:
        file_names=pickle.load(file)
    longest=0.1998710064553202
    Trial=Simulation(num_of_envs=24, render=True, agent="bodyv8", longest=longest)
    Trial.start(time_steps=350, episodes=500, discount=.9, minibatch_len=8, lam=.9, save_interval=10, save_dir='PPO Trial 1', params=None)