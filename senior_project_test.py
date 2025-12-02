import pybullet as p
import time
import pybullet_data
from xacrodoc import XacroDoc
import os

def read_xacro(file_name: str) -> str:
    current_file = file_name

    doc = XacroDoc.from_file(f"{current_file}.urdf.xacro")

    urdf_string = doc.to_urdf_string()

    with open(f"{current_file}.urdf", "w") as f:
            f.write(urdf_string)
            
    return file_name

current_robot=read_xacro("dogv1")
current_environment="plane"

physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
planeId = p.loadURDF(f"{current_environment}.urdf")
startPos = [0,0,1.25]
startOrientation = p.getQuaternionFromEuler([0,0,0])
dog = p.loadURDF(f"{current_robot}.urdf",startPos, startOrientation)
p.changeDynamics(bodyUniqueId= dog, linkIndex=-1, lateralFriction=1)
maxForce = 50000
targetVel=12



f_slider = p.addUserDebugParameter("Force",-20,20,10)

numOfJoints = p.getNumJoints(dog)
jointIds = range(numOfJoints)

print("here", numOfJoints, jointIds)
joints_dict={}
for num in range (numOfJoints):
    if p.getJointInfo(dog, num)[2] != p.JOINT_FIXED:
        joints_dict[p.getJointInfo(dog, num)[0]]=(p.addUserDebugParameter(str(p.getJointInfo(dog, num)[1]),-1.5,1.5,0))

print(joints_dict)
positions={}

while True:
    
    for i in joints_dict.keys():
        positions[i] = (p.readUserDebugParameter(joints_dict[i]))
    
    p.setJointMotorControlArray(dog, joints_dict.keys(), controlMode=p.POSITION_CONTROL, targetPositions=positions.values())
    p.stepSimulation()
    
    time.sleep(1./240.)
    
    

p.disconnect()
