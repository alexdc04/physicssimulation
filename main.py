from pybullet_utils import bullet_client as bc
import pybullet, time, pybullet_data, torch, random, bisect, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
#can also connect using different modes, GUI, SHARED_MEMORY, TCP, UDP, SHARED_MEMORY_SERVER, GUI_SERVER
#pgui = bc.BulletClient(connection_mode=pybullet.GUI)

POSITION_CONTROL=0 
TORQUE_CONTROL=1
VELOCITY_CONTROL=2
device=torch.device('cuda')
plt.ion()
plt.style.use('fivethirtyeight')

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
            log_std=torch.clamp(self.log_std(x), -20, 2)
            return (mean , log_std)
        else:
            return self.value(x)
        
    def save(self, dir: str, name):
        torch.save(self.state_dict(), f'{dir}/{name}.pth')

class Simulation():
    def __init__(self, num_of_envs, render, agent, longest=0):
        self.envs= {}
        self.longest_dist=longest
        self.pos=[0, 0, .21]
        self.connect(num_of_envs, render, agent)
        self.joints={self.envs[0].getJointInfo(1, joint)[1]:self.envs[0].getJointInfo(1, joint)[0] 
                        for joint in range(self.envs[0].getNumJoints(1)) 
                        if self.envs[0].getJointInfo(1, joint)[2] != 4}
        self.num_envs=num_of_envs
        
        if render: 
            self.num_envs+=1
        self.render=render
        
        for env in self.envs.values():
            env.changeDynamics(1, self.joints[b'left_ankle'], lateralFriction=1.0)
            env.changeDynamics(1, self.joints[b'right_ankle'], lateralFriction=1.0)
            
        self.episode=[]
        self.rewards=[]
        self.p_loss=[]
        self.v_loss=[]
        self.dists_absmax=[]
        self.dists_epmax=[]
        self.avg_survival=[]
        self.max_survival=[]
        self.running_best=[]
        
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
            
            p.addUserDebugLine([self.longest_dist,-1,0], [self.longest_dist,1,0], [0, 1, 0], 3)

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
        right_knee_touching=(self.get_collision(env_idx=env_idx, joint_name=b'right_hip_joint'))
        left_knee_touching=(self.get_collision(env_idx=env_idx, joint_name=b'left_hip_joint'))
        
        alive_bonus=3
        punish=-3
        
        if left_foot_touching:
            alive_bonus+=.5
            
        if right_foot_touching:
            alive_bonus+=.5
            
        if left_knee_touching or right_knee_touching:
            alive_bonus-=4
            punish-=4
            
        if dist>self.running_best:
            self.running_best=dist
            if dist>self.longest_dist:
                #print(f'New Record! {dist}')
                if self.render: 
                    self.envs[-1].removeAllUserDebugItems()
                    self.envs[-1].addUserDebugLine([dist,-1,0], [dist,1,0], [0, 1, 0], 3)
                self.longest_dist=dist
                dist*=2.5
        
        if dist>.15:
            alive_bonus*=3
            punish+=1
            if dist>.25:
                alive_bonus*3
                punish+=1
            
        if episode>=75:
            punish*1.25
            alive_bonus*.90
            dist*=.86
            if episode>=150:
                punish*1.25
                alive_bonus*.90
                dist*=.86
                
                
        if z < .19:
            return punish, True
        else:
            return dist*1.5+(alive_bonus)*.85, False
    
    def worm_reward(self, env_idx, episode):
        dist, h, p=self.envs[env_idx].getLinkState(1, self.joints[b'bend_low_torso'])[0]
        return dist, False
    
    def start(self, time_steps, episodes, discount, minibatch_len, lam, save_interval, save_dir, params, updates):
        s0=self.get_state(0)
        rb=Replay_Buffer(capacity=time_steps)
        policy=NeuralNetwork(num_states=len(np.concatenate(s0)), num_actions=len(s0[0]), hidden_dims=128, policy=True).to(device)
        value=NeuralNetwork(num_states=len(np.concatenate(s0)), num_actions=1, hidden_dims=128, policy=False).to(device)
        fig, ax = plt.subplots(2, 3, figsize=(13.5, 6.75))
        longest_alive=np.zeros(self.num_envs)
        running_alive_count=np.zeros(self.num_envs)
        ax[0][2].set_visible(False)
        if params:
            policy.load_state_dict(params['policy'])
            value.load_state_dict(params['value'])
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        value_optimizer  = torch.optim.Adam(value.parameters(),  lr=1e-3)
        
        for episode in range(episodes):
            self.running_best=0
            avg_reward=0
            running_alive_count=np.zeros(self.num_envs)
            for idx in self.envs.keys():
                self.reset_agent(idx)
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
                    np.add.at(running_alive_count, idx, 1)
                    r, fail= self.reward(idx, episode)
                    r_list[idx] = r

                    if fail:
                        if running_alive_count[idx]>longest_alive[idx]:
                            longest_alive[idx]=running_alive_count[idx]
                        running_alive_count[idx]=0
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
                rb.buffer[t].append(torch.tensor(gae.detach()).float())
                
            # returns
            for t in range(time_steps):
                rb.buffer[t].append((rb.buffer[t][5] + rb.buffer[t][4].squeeze(-1)).detach())
            
            sample = rb.sample(minibatch_len)
            for x in range(updates):
                for i, x in enumerate(sample):
                    print(i, type(x[1]), getattr(x[1], "shape", None))
                try:  
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
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - eps, 1 + eps) * advantages).mean()
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

                    value_pred = value(states).squeeze(-1)
                    value_loss = F.mse_loss(value_pred, returns.float())
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    value_optimizer.step()    
                except:
                    print('error in cycle')
                    try:
                        policy_loss=self.p_loss[episode-1]
                        value_loss=self.v_loss[episode-1]
                    except:
                        policy_loss=torch.tensor(0)
                        value_loss=torch.tensor(0)
                    
                try: 
                    policy_loss=policy_loss.detach()
                    value_loss=value_loss.detach()
                except:
                    policy_loss=policy_loss
                    value_loss=value_loss
                
            if episode%save_interval==0:
                plt.savefig(fname=f'trial 8')
                save={'policy':policy.state_dict(), 'value':value.state_dict()}
                with open(f'{save_dir}\\weights{episode}.pkl', 'wb') as file:
                    pickle.dump(save, file)
            
            self.episode.append(episode)
            self.rewards.append((avg_reward)/self.num_envs)
            self.p_loss.append(float(policy_loss))
            self.v_loss.append(float(value_loss))
            self.dists_absmax.append(self.longest_dist)
            self.dists_epmax.append(self.running_best)
            self.avg_survival.append(running_alive_count.mean())
            self.max_survival.append(longest_alive.mean())
            ax[1][0].cla()
            ax[1][1].cla()
            ax[0][1].cla()
            ax[0][0].cla()
            ax[1][2].cla()
            ax[1][2].cla()
            ax[1][0].plot(self.episode, self.rewards, label='Reward', color="green")
            ax[0][0].plot(self.episode, self.p_loss, label='Policy Loss', color="blue")
            ax[0][1].plot(self.episode, self.v_loss, label='Value Loss', color="red")
            ax[1][1].plot(self.episode, self.dists_absmax, label='Furthest Max Distance', color="orange")
            ax[1][1].plot(self.episode, self.dists_epmax, label='Furthest Episode Distance', color="purple")
            ax[1][2].plot(self.episode, self.avg_survival, label='Average Survival', color="magenta")
            ax[1][2].plot(self.episode, self.max_survival, label='Max Survival', color="cyan")
            ax[1][0].legend(loc='best')
            ax[1][1].legend(loc='best')
            ax[0][1].legend(loc='best')
            ax[0][0].legend(loc='best')
            ax[1][2].legend(loc='best')
            ax[1][2].legend(loc='best')
            ax[1][0].set_xlabel('Epoch')
            ax[1][0].set_ylabel('Value')
            ax[0][0].set_xlabel('Epoch')
            ax[0][0].set_ylabel('Loss')
            ax[0][1].set_xlabel('Epoch')
            ax[0][1].set_ylabel('Loss')
            ax[1][1].set_xlabel('Epoch')
            ax[1][1].set_ylabel('Meters')
            ax[1][1].set_xlabel('Epoch')
            ax[1][1].set_ylabel('Meters')
            ax[1][2].set_xlabel('Epoch')
            ax[1][2].set_ylabel('Time Steps')
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
            #print(f'Avg-Reward:{avg_reward/self.num_envs}\nValue Loss:{value_loss}\nPolicy Loss:{policy_loss}\nLongest Dist:{self.longest_dist}')        
        
        


if __name__=='__main__':
    weights='\\sample_weights.pkl'
    with open(weights, 'rb') as file:
        file_names=pickle.load(file)
    
    longest=2.65

    Trial=Simulation(num_of_envs=24, render=False, agent="bodyv8", longest=longest)
    Trial.start(time_steps=350, episodes=2000, discount=.99, minibatch_len=100, lam=.97, save_interval=5, save_dir='New_Trial', params=file_names, updates=5)
    
    