import pybullet as p
import numpy as np
import time
import cv2

# Or directly
from environment import PyBulletContinuousEnv
env = PyBulletContinuousEnv()

# Inizializzo l'ambiente
env.reset()

done = False
turn=0
forward=0
backward=0

# dopo 1000 step done diventa True
while not done:
    #time.sleep(1./240.)
    p.setGravity(0,0,-10)
    keys = p.getKeyboardEvents()

# comandi da tastiera per l'expert
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
                    backward=20
            if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
                    backward=0

    # la riga 161 del dagger.py deve essere sostituita con questa
    action = np.array([turn, forward, backward]).astype('float32')
    #obs = env.get_observation()
    obs, reward, done = env.step(action)
    image = env.visualization_image()
    cv2.imshow("Camera", image)
    cv2.waitKey(1) 

env.close()
