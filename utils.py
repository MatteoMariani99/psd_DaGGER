import numpy as np

# LEFT     = 1
# RIGHT    = 2
# ACC      = 3
# BRAKE    = 4
# STRAIGHT = 0

# action_to_id_dict = {LEFT     : np.array([-1.0, 0.0, 0.0]),
#                      RIGHT    : np.array([+1.0, 0.0, 0.0]),
#                      ACC      : np.array([0.0, +1.0, 0.0]),
#                      BRAKE    : np.array([0.0, 0.0, +0.2]),
#                      STRAIGHT : np.array([0.0, 0.0, 0.0 ])}

# CUTOFF = 84 # for pixels                     
# def rgb2yuv(rgb):
#     """ 
#     this method converts rgb images to grayscale.
#     """
#     trans_matrix = np.array([[0.299,-0.16874,0.5],
#                              [0.587,-0.33126,-0.41869],
#                              [0.114,0.5,-0.08131]])
#     yuv = np.dot(rgb[...,:3], trans_matrix)
#     yuv[:,:,:1]+=128.0
#     return yuv.astype('float32') 

# def action_to_id(y_samples):
#     """
#     this method turns samples of actions into an id.
#     y_samples should be of size (NUM_SAMPLES, 3)
#     """

#     y_ids = np.zeros((y_samples.shape[0]))

#     for key, var in action_to_id_dict.items():
#       curr_idxs = np.all(y_samples == var, axis=1)
#       y_ids[curr_idxs] = key

#     return y_ids

# def wrap2pi(angle):
#     return (angle + np.pi) % (2 * np.pi) - np.pi


# def rect_to_polar_relative(goal):
# 	"""
# 	Funzione usata per la trasformazione in coordinate polari del goal
	
# 	Parameters:
# 	- robot_id: id del robot
	
# 	Returns:
# 	- r: raggio, ovvero la distanza tra robot e goal
# 	- theta: angolo di orientazione del robot rispetto al goal
# 	"""
	
# 	# posizioe del goal
# 	goal = goal
	
# 	# calcolo la posizione correte del robot specificato
# 	robPos, car_orientation = p.getBasePositionAndOrientation(turtle)

# 	# calcolo l'angolo di yaw
# 	_,_,yaw = p.getEulerFromQuaternion(car_orientation)
	
# 	# Calculate the polar coordinates (distance, angle) of the vector
# 	vector_to_goal = np.array([goal[0] - robPos[0], goal[1] - robPos[1]])
# 	r = np.linalg.norm(vector_to_goal)
# 	theta = wrap2pi(np.arctan2(vector_to_goal[1], vector_to_goal[0])-wrap2pi(yaw))
# 	return r, theta


# def choosePositionAndIndex(position,index):
# 	if len(position!=0):
# 		for i,j in zip(range(len(position)),position):
# 			r, yaw_error = rect_to_polar_relative(j[:2])
# 			vel_ang = p_control(yaw_error)
# 			print(f"steer {vel_ang} - distance {r}")
# 			if vel_ang < 1.5:
# 				positionToStart = j[:2]
# 				indexToStart = index[i]
# 				done = True
# 			else:
# 				positionToStart = []
# 				indexToStart = None
# 				done = False
# 	else:
# 		positionToStart = []
# 		indexToStart = None
# 		done = False
# 	return positionToStart, indexToStart, done