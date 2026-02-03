"""
<Train.py>: Program will train model based on environment given.

<Detailed description/explanation of what the script does and how it works>

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



if __name__ == "__main__":
    environment_folder="Basic Walking"
    session_no=1
    model=""
    
    with open(f'{environment_folder}/Session_{session_no}.json', 'rb') as f:
        data = msgspec.json.load(f)
        
    print(data)

