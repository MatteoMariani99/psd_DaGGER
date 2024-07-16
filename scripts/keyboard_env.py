import pybullet as p
import cv2

#? Import ambienti
from environment_cones import ConesEnv
from environment_road import RoadEnv

cones = True
if cones:
	env = ConesEnv()
else:
    env = RoadEnv()

# funzione che attende l'input da parte dell'utente
def wait():
    print("Premere qualsiasi tasto per iniziare a guiare!")
    _ = input()



if __name__ == "__main__":    

	done = False
	turn=0
	forward=0
	backward=0

	env.reset()
	print("""Comandi per la guida:
	---------------------------
	left (arrow left)
	right (arrow right)
	up (arrow up)
	back (arrow back)
 	---------------------------
					""")
		
	wait()

	while not done:
		# Estrapolazione immagine a colori
		color_rgb = env.getCamera_image()
		
		# Lettura tasti premuti della tastiera
		keys = p.getKeyboardEvents()

		# comandi da tastiera per guidare
		for k,v in keys.items():
			if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
				turn = -0.5
			if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
				turn = 0
			if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
				turn = 0.5
			if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
				turn = 0

			if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
				forward=10
			if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
				forward=0
			if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
				forward=-10
			if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
				forward=0
	
	
		image,_,_ = env.step([turn,forward])
		cv2.imshow("YOLOv8 Tracking", cv2.cvtColor(color_rgb,cv2.COLOR_BGR2RGB))
	
		# Display the annotated frame
		#cv2.imshow("YOLOv8 detect", annotated_frame)
		#cv2.imshow('IMAGE', img)
		cv2.waitKey(1)

	env.close()
