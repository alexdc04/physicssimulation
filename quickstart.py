import torch
from torch.utils.data import DataLoader, Dataset


t1=[(torch.arange(2, dtype=torch.float32), torch.arange(3, dtype=torch.float32), x, x*2) for x in range(0, 5)]

class ReplayBuffer(Dataset):
    def __init__(self, data):
        self.data=data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

t2=ReplayBuffer(t1)
test= DataLoader(t2, batch_size=3, shuffle=True)

data=[x for x in test]
d=data[0]

for x in range(0, 3):
    print(f"{d[0][x], d[1][x], d[2][x], d[3][x]}\n")