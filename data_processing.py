from pathlib import Path
from xacrodoc import XacroDoc
import pickle


def read_xacro(file_name: str) -> str:
    current_file = file_name

    doc = XacroDoc.from_file(f"models/xacro/{current_file}.urdf.xacro")

    urdf_string = doc.to_urdf_string()

    with open(f"models/raw/{current_file}.urdf", "w") as f:
            f.write(urdf_string)
            
    return "models/raw/" + file_name + ".urdf"

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
    
class AgentNotFound(Exception):
    """A custom exception class for specific errors."""
    pass

class InactiveSim(Exception):
    """A custom exception class for specific errors."""
    pass