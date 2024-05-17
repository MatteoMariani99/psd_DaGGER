import pybullet as p
import time
import pybullet_data
import numpy as np
import csv
import pandas as pd

# velocità rispetto al frame macchina
def getVelocity():
        linearVelocity, angularVelocity = p.getBaseVelocity(turtle)
        base_pose =  p.getBasePositionAndOrientation(turtle)

        # Converto le velocità angolari e lineari in local frame
        rotation_matrix = p.getMatrixFromQuaternion(base_pose[1])  # Rotation matrix from quaternion
        rotation_matrix = np.reshape(rotation_matrix, (3, 3))

        # Velocità lineare nel frame locale
        local_linear_velocity = np.dot(np.linalg.inv(rotation_matrix), np.array(linearVelocity))

        # Velocità angolare nel frame locale
        local_angular_velocity = np.dot(np.linalg.inv(rotation_matrix), np.array(angularVelocity))

        # Extract the velocity values
        linear_velocity = np.array(local_linear_velocity)[0] # x velocity
        angular_velocity = np.array(local_angular_velocity)[2] # z angular

        return linear_velocity, angular_velocity


# Path to the CSV file
csv_file_path = 'position_data.csv'



p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=30, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[8,0,0])
offset = [0,0,0]
p.loadURDF("plane.urdf")
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [0,0,.3])
turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-15,-11,.3])
#track = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)
p.loadURDF("cube/marble_cube.urdf",[-3,2,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,1,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,3,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,4,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,0,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,-1,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,-2,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,-3,0],useFixedBase = True)

# horizontal
p.loadURDF("cube/marble_cube.urdf",[3,5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[2,5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[4,5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[1,5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[5,5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[6,5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[7,5,0],useFixedBase = True)

# vertical
p.loadURDF("cube/marble_cube.urdf",[8,7,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[8,6,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[8,8,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[8,5,0],useFixedBase = True)

# vertical
p.loadURDF("cube/marble_cube.urdf",[-8,7,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-8,6,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-8,8,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-8,5,0],useFixedBase = True)
p.setRealTimeSimulation(1)



# car_velocity_x, car_angular_velocity_z = getVelocity()

# _,_,yaw = p.getEulerFromQuaternion(car_orientation)


# sphere_color = [1, 0, 0,1]  # Red color
# sphere_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0)
# for i in range (1,10):
#         p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape,basePosition=[1,i,2])
#p.changeVisualShape(sphere_body, -1, rgbaColor=sphere_color)

# for j in range (p.getNumJoints(turtle)):
# 	print(p.getJointInfo(turtle,j))
forward=0
turn=0
start_time = time.time()

with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header if the file is new/empty
    file.seek(0)
    if file.tell() == 0:
        writer.writerow(['X', 'Y'])
    try:
        while True:
                # Get the current time and position
                #timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                car_position, _ = p.getBasePositionAndOrientation(turtle)
                x,y = car_position[:2]
                # Write the data to the CSV file
                writer.writerow([x, y])
                p.setGravity(0,0,-10)
                #time.sleep(1./240.)
                keys = p.getKeyboardEvents()
                leftWheelVelocity=0
                rightWheelVelocity=0
                speed=10

                for k,v in keys.items():

                        if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                                turn = 0.5
                        if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
                                turn = 0
                        if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                                turn = -0.5
                        if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
                                turn = 0

                        if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                                forward=20
                        if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
                                forward=0
                        if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                                forward=-20
                        if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
                                forward=0

                #print(forward)ù
                # F10 RACECAR
                # p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=forward)
                # p.setJointMotorControl2(turtle,3,p.VELOCITY_CONTROL,targetVelocity=forward)
                # p.setJointMotorControl2(turtle,12,p.VELOCITY_CONTROL,targetVelocity=forward)
                # p.setJointMotorControl2(turtle,14,p.VELOCITY_CONTROL,targetVelocity=forward)
                # p.setJointMotorControl2(turtle,0,p.POSITION_CONTROL,targetPosition=-turn)
                # p.setJointMotorControl2(turtle,2,p.POSITION_CONTROL,targetPosition=-turn)

                # SIMPLE CAR
                p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=forward)
                p.setJointMotorControl2(turtle,3,p.VELOCITY_CONTROL,targetVelocity=forward)
                p.setJointMotorControl2(turtle,4,p.VELOCITY_CONTROL,targetVelocity=forward)
                p.setJointMotorControl2(turtle,5,p.VELOCITY_CONTROL,targetVelocity=forward)
                p.setJointMotorControl2(turtle,0,p.POSITION_CONTROL,targetPosition=-turn)
                p.setJointMotorControl2(turtle,2,p.POSITION_CONTROL,targetPosition=-turn)

                print(f"{x}, {y}")
                      
                # Wait for one second
                time.sleep(1)
    except KeyboardInterrupt:
        # Handle the interrupt (e.g., user presses Ctrl+C)
        print("Stopping data collection.")

        #while (True):
                
                


        print("--- %s seconds ---" % (time.time() - start_time))

