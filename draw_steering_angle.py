import cv2
import numpy as np


class SteeringWheel():
    def __init__(self, background_image):
        self.background_image = background_image
        self.wheel_path = 'steer_prova.png'

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
        return self.background_image


    def draw_steering_wheel_on_image(self, steer_angle, position):
        # Load background and steering wheel images
        #background = self.load_image(background_path)
        steering_wheel = self.load_image(self.wheel_path)
        
        # Rotate steering wheel based on steer_angle
        rotated_wheel = self.rotate_image(steering_wheel, steer_angle)
        
        # Overlay the rotated steering wheel on the background
        result_image = self.overlay_image(rotated_wheel, position[0], position[1])
        
        return result_image
    
    def draw_vertical_bar(self, image, value):
        """Draw a vertical bar on the image."""
        # Calculate the current height of the bar based on the value
        current_height = int((10/value))
        
        pts = np.array([[400, 30+current_height], [440, 30+current_height],
                [440, 60], [400, 60]],
               np.int32)
 
        pts = pts.reshape((-1, 1, 2))
        # Draw the bar
        cv2.fillPoly(image, pts=[pts], color=(255, 255, 255))
        #cv2.rectangle(image, (400, 10), (410, 10-current_height), color, -1)
        
        return image

    def update_frame_with_bar(self, velocity):
        """Update the frame with the animated vertical bar based on the velocity."""
        frame = self.background_image.copy()
        frame = self.draw_vertical_bar(frame, velocity)
        return frame


# Example usage
#background_path = 'wheel.jpg'



# while 1:
#     for i in range (0,100,1):
#         steer_angle = i  # Example steering angle
#         position = (100, 100)  # Position to place the steering wheel on the background

#         # Draw steering wheel on image
#         result_image = draw_steering_wheel_on_image(background_path, wheel_path, steer_angle, position)

#         # Display the result
#         cv2.imshow('Steering Wheel Animation', result_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

