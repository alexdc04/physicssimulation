from pathlib import Path
from xacrodoc import XacroDoc
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import json
import io
import pickle
import time

def read_xacro(file_name: str) -> str:
    current_file = file_name

    doc = XacroDoc.from_file(f"models/xacro/{current_file}.urdf.xacro")

    urdf_string = doc.to_urdf_string()

    with open(f"models/raw/{current_file}.urdf", "w") as f:
            f.write(urdf_string)
            
    return "models/raw/" + file_name + ".urdf"

class DriveConnection():
    def __init__(self):
        self.drive=self.connect_to_drive()
        self.types=json.load(open('mime_types.json'))
        print("Loading Directories.")
        self.main_dirs=self.query(conditions=[f"mimeType='{self.types.get('folder')}'"])
        print("Finished loading directories.\n")
    
    def connect_to_drive(self):
        """Returns connection for a drive connection. Correct yaml settings or secret is required."""
        gauth = GoogleAuth() 
        gauth.LocalWebserverAuth()
        return GoogleDrive(gauth)

    def get_dir_id(self, dir: str):
        return self.main_dirs.get(f'{dir}')
    
    def query(self, dir='root', conditions=[]) -> dict:
        """
        Queries database. By default it will give everything in the root dir.\n
        Conditions should be in the following format;\n
        "["condition = 'value'", ...]'"\n
        This returns the requested data as {title:id}
        """
        query=''
        if conditions:
            for x in conditions:
                query+= f"and {x}"
        old=time.time()
        print(f"Beginning query with extra conditions: {query}")
        file_list = self.drive.ListFile({'q': f"'{dir}' in parents and trashed=false {query}"}).GetList()
        print(f"Finished query. Time Elapsed: {time.time()-old} seconds.\n")
        return {x['title']: x['id'] for x in file_list}
    
    def get(self, dir='root', conditions=[]) -> list:
        """
        Extension of query. If desired, files are converted from drive to correct format.\n
        """
        location=self.get_dir_id(dir)
        values=self.query(location, conditions)
        old=time.time()
        print(f"Beginning conversion for {len(conditions)} files.")
        out=[]
        for value in list(values.values()):
            temp = (self.drive.CreateFile({'id':value}))
            temp.FetchContent()
            out.append(pickle.loads(temp.content.read()))
        print(f"Finished Conversion. Time Elapsed: {time.time()-old} seconds.\n")
        return out
    
    def save(self, data: object, name: str, dir='root'):
        """ 
        Saves data to drive. Default directory is root.
        """
        location=self.get_dir_id(dir)
        old=time.time()
        print(f"Beginning Data Insertion into {dir}")
        file=self.drive.CreateFile({'title': name, "parents": [{"id": f"{location}"}]})
        file.content = io.BytesIO(pickle.dumps(data))
        file.Upload()
        print(f"Finished Insertion. Time Elapsed: {time.time()-old} seconds.\n")
        
    def exists(self, name, dir: str):
        location=self.get_dir_id(dir)
        if self.query(dir=f'{location}', conditions=[f"title='{name}_rm'"]):
            return True
        else:
            return False
    

def load_session_data(hp_name: str, model_name: str, conn: DriveConnection):
    out={}
    out['model_name']=model_name
    out['hp']=json.load(open(f'Hyperparameters/{hp_name}'))
    out['replay_memory']=conn.get(dir='Replay_Memory', conditions=[f"title='{model_name}_rm'"])
    out['policy_net_parameters']=conn.get(dir='DQN_Parameters', conditions=[f"title='{model_name}_pn_weights'"])
    out['target_net_parameters']=conn.get(dir='DQN_Parameters', conditions=[f"title='{model_name}_tn_weights'"])
    out['stats']=conn.get(dir='Stats', conditions=[f"title='{model_name}_stats'"])
    return out

def save_session_data(data: dict, conn: DriveConnection):
    name=data.get('model_name')
    conn.save(data.get('replay_memory'), name+'_rm', dir='Replay_Memory')
    conn.save(data.get('policy_net_parameters'), name+'_pn_weights', dir='DQN_Parameters')
    conn.save(data.get('target_net_parameters'), name+'_tn_weights', dir='DQN_Parameters')

test=DriveConnection()

print(test.exists('firs', 'Replay_Memory'))