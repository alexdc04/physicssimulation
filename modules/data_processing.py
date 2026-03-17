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


from pathlib import Path
from xacrodoc import XacroDoc
import json
import io
import pickle
import time
import string
import random
def read_xacro(file_name: str) -> str:
    """Loads xacro model into string.

    Args:
        file_name: Name of model.

    Returns:
        Raw xacro string
    """
    
    current_file = file_name

    doc = XacroDoc.from_file(f"models/xacro/{current_file}.urdf.xacro")

    urdf_string = doc.to_urdf_string()

    with open(f"models/raw/{current_file}.urdf", "w") as f:
            f.write(urdf_string)
            
    return "models/raw/" + file_name + ".urdf"

def load_json(dir_name: str, file_name: str) -> dict:
    """Loads a dict from JSON file.

    Args:
        dir_name: Name of directory.
        file_name: Name of file.

    Returns:
        Returns a dict from given JSON.
    """
    try:
        with open(f'{dir_name}/{file_name}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
    
    return data

def save_session_data():
    print("placeholder")
    
def generate_random_key(length=16):
        """Generates a general-purpose pseudo-random key of a specified length."""
        # Define the possible characters for the key
        characters = string.ascii_letters + string.digits
        # Use random.choices with join to generate the string efficiently
        key = ''.join(random.choices(characters, k=length))
        return key

def general_save(data: object, dir_path: str):
    key=dir_path+srt(int(time.time()))+'.pkl'
    
    with open(key, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
        
def general_load(dir_path: str):

    with open(dir_path, 'rb') as f:
        return pickle.load(f)
        
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
    pol=None
    target=None
    
    return data["Physics_Settings"], data['Situation'], data['Hyperparameters']