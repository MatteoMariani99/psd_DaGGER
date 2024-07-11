import yaml
import numpy as np
import track_creator
import cv2

left_position = []
right_position = []

# OK
# tutti scale 0.5 eccetto il 6,7,9 con scale 0.6

# per il 4 vanno invertite le coordinate del centerLine

# TESTING: 6,7,9

track_number = 9

print(track_number)

with open(f'/home/tazio/Desktop/psd_DaGGER/world/fsd_racetrack_dataset/dataset/boundaries_{track_number}.yaml', 'r') as file:
    bound = yaml.safe_load(file)
   
left_cones_id = bound['left']
right_cones_id = bound['right']

with open(f'/home/tazio/Desktop/psd_DaGGER/world/fsd_racetrack_dataset/dataset/cone_map_{track_number}.yaml', 'r') as file:
    position = yaml.safe_load(file)
 
for left in left_cones_id:
    left_position.append(position[left])   

for right in right_cones_id:
    right_position.append(position[right])
   
   


# Me ne frego del segno

rd_area = cv2.arcLength(np.array(left_position).astype(np.float32), closed=True) - \
    cv2.arcLength(np.array(right_position).astype(np.float32), closed=True)
            
# Avg length
rd_len = (cv2.contourArea(np.array(right_position).astype(np.float32)) + \
        cv2.contourArea(np.array(left_position).astype(np.float32)))/2

rd_w = abs(rd_area/ rd_len)
#print(rd_w)
# Valore desiderato (in metri) diviso valore attuale
scale_factor = 0.6
#0.006/rd_w
print(rd_w)

scaled_inner = track_creator.scale_track(np.array(right_position), scale_factor)
scaled_outer = track_creator.scale_track(np.array(left_position), scale_factor)

#print(scaled_inner)
    
np.savetxt(f"/home/tazio/Desktop/psd_DaGGER/world/cones_position/coneIN_track{track_number}.csv", scaled_inner,  
            delimiter = ",")
np.savetxt(f"/home/tazio/Desktop/psd_DaGGER/world/cones_position/coneOUT_track{track_number}.csv", scaled_outer,  
            delimiter = ",")



sdf_content = track_creator.generate_sdf(scaled_inner, scaled_outer)



with open(f"/home/tazio/Desktop/psd_DaGGER/world/track/track{track_number}.sdf", "w") as f:
    f.write(sdf_content)