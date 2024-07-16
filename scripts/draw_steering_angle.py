import cv2
import numpy as np

# funzione utile a disegnare il volante e la barra verticale
class SteeringWheel():
    def __init__(self, background_image):
        self.background_image = background_image
        self.wheel_path = 'docs/steer.png'

    def load_image(self,image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (50, 50))
        if image is None:
            raise ValueError(f"Image at path '{image_path}' could not be loaded.")
        return image

    def rotate_image(self,image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        return rotated

    def overlay_image(self, overlay, x, y):
        (h, w) = overlay.shape[:2]
    
        # If the background is grayscale, convert it to BGR
        if len(self.background_image.shape) == 2 or self.background_image.shape[2] == 1:
            self.background_image = cv2.cvtColor(self.background_image, cv2.COLOR_GRAY2BGR)
        
        # Extract the region of interest (ROI) from the background
        roi = self.background_image[y:y+h, x:x+w]

        # Separate the color and alpha channels of the overlay
        overlay_color = overlay[:, :, :3]
        if overlay.shape[2] == 4:  # has alpha channel
            alpha_mask = overlay[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + overlay_color[:, :, c] * alpha_mask
        else:
            roi[:, :, :3] = overlay_color
        
        # Place the modified ROI back into the background image
        self.background_image[y:y+h, x:x+w] = roi


    def draw_steering_wheel_on_image(self, steer_angle, position):
        # Load background and steering wheel images
        steering_wheel = self.load_image(self.wheel_path)
        
        # Rotate steering wheel based on steer_angle
        rotated_wheel = self.rotate_image(steering_wheel, steer_angle)
        
        # Overlay the rotated steering wheel on the background
        self.overlay_image(rotated_wheel, position[0], position[1])
        
    
    def draw_vertical_bar(self, image, value):
        """Draw a vertical bar on the image."""
        
        # Calculate the current height of the bar based on the value
        if value!=0 and value<=10:
            current_height = (10-value)*5
        elif value > 10:
            current_height = 0
        else:
            current_height = 50
        
        pts = np.array([[300, 5+current_height], [340, 5+current_height],
                [340, 55], [300, 55]],
               np.int32)
 
        pts = pts.reshape((-1, 1, 2))
        
        # Draw the bar
        cv2.fillPoly(image, pts=[pts], color=(255, 255, 255))
        cv2.putText(image, "10", (280,10), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.3, (255,255,255), 1, cv2.LINE_AA) 
        cv2.putText(image, "0", (285,58), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.3, (255,255,255), 1, cv2.LINE_AA)
        
        return image

    def update_frame_with_bar(self, velocity):
        """Update the frame with the animated vertical bar based on the velocity."""
        frame = self.background_image.copy()
        frame = self.draw_vertical_bar(frame, velocity)
        return frame

