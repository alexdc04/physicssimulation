import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random 
import pandas as pd
from collections import deque
# First why dont we start with a smaller example.
# Our map will be a 8x8 grid with a bad squares, empty squares, and a goal square.
grid=np.zeros(shape=(8, 8))
bounds={x: 0 for x in range(0, 8)}
goal=np.array([7, 7])
found=False
print(bounds)
for row in grid:
    
    row[random.randint(0, 7)]=-1

grid[0][0]=1 # Agent
grid[7][7]=2 # Goal

curr_pos=np.array([0, 0])
moved=[np.array([0, 0])]
moves = {
    'left': (0, -1),
    'right': (0, 1),
    'up': (-1, 0),
    'down': (1, 0)
}
dirs=list(moves.keys())

def move(curr_pos, dir: str):
    
    last=curr_pos
    new=curr_pos + moves[dir]
    
    if (new[0] in bounds and new[1] in bounds) and not grid[new[0], new[1]] == -1:
        grid[last[0], last[1]] = 0
        grid[new[0], new[1]] = 1
        return new
    else:
        print("Cant Move")
        return last

while not found:
    for x in range(25):
        curr_pos = move(curr_pos,(random.sample(dirs, 1)[0]))
        moved.append(curr_pos)
        if (curr_pos[0] == 7 and curr_pos[1] == 7) :
            found=True
            break

print(moved)
print(grid)
























# class DQN(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(DQN, self).__init__()
#         self.test_stack = nn.Sequential(
#             nn.Linear(state_size, 50),
#             nn.ReLU(),
#             nn.Linear(50, 50),
#             nn.ReLU(),
#             nn.Linear(50, action_size),
#             nn.Softmax()
#         )
#     def forward(self, x):
#         return self.test_stack(x)
    


# data=torch.tensor(grid.flatten()).float()



# # Hyperparameters
# gamma = 0.99             
# epsilon = 1.0           
# epsilon_min = 0.01
# epsilon_decay = 0.995
# learning_rate = 0.001
# batch_size = 64
# memory_size = 10000
# state_size=64
# action_size=4

# memory = deque(maxlen=memory_size)
# policy_net = DQN(state_size, action_size)
# target_net = DQN(state_size, action_size)
# target_net.load_state_dict(policy_net.state_dict())  
# target_net.eval()

# optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
# loss_fn = nn.MSELoss()



