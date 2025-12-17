import sqlite3
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import json
import torch
import io
import pickle
import time

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
        values=self.query(dir, conditions)
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
        old=time.time()
        print(f"Beginning Data Insertion.")
        file=self.drive.CreateFile({'title': name, "parents": [{"id": f"{dir}"}]})
        file.content = io.BytesIO(pickle.dumps(data))
        file.Upload()
        print(f"Finished Insertion. Time Elapsed: {time.time()-old} seconds.\n")



conn=DriveConnection()
test=torch.tensor([0, 0, 0])
conn.save(data=test, name='Insert Test', dir='root')
want = conn.get(dir='root', conditions=["title='image_test'"])
print(want)

