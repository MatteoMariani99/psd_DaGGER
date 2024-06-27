import numpy as np
import matplotlib.pyplot as plt

# Define the two circles
circle1_center = (0, 0)
circle1_radius = 38
circle2_center = (0, 0)
circle2_radius = 40

# Number of cones
num_cones = 80

# Calculate points on the circumference of circle 1
theta = np.linspace(0, 2 * np.pi, num_cones, endpoint=False)
circle1_points = np.array([(circle1_center[0] + circle1_radius * np.cos(t), 
                   circle1_center[1] + circle1_radius * np.sin(t)) for t in theta])

# Calculate points on the circumference of circle 2
circle2_points = np.array([(circle2_center[0] + circle2_radius * np.cos(t), 
                   circle2_center[1] + circle2_radius * np.sin(t)) for t in theta])


print(circle2_points.tolist())
print(circle1_points.tolist())

# Plot the circles and the cones
plt.figure(figsize=(8, 8))
circle1 = plt.Circle(circle1_center, circle1_radius, color='blue', fill=False, linestyle='--')
circle2 = plt.Circle(circle2_center, circle2_radius, color='red', fill=False, linestyle='--')
plt.gca().add_artist(circle1)
plt.gca().add_artist(circle2)

# Plot cones
for point in circle1_points:
    plt.plot(point[0], point[1], 'bo')  # Blue cones for circle 1

for point in circle2_points:
    plt.plot(point[0], point[1], 'ro')  # Red cones for circle 2





import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def create_ellipse(a, b, center=(0, 0), num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + a * np.cos(angles)
    y = center[1] + b * np.sin(angles)
    return x, y

def smooth_path(x, y, num_points=1000):
    tck, u = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, num_points)
    smooth_path = splev(unew, tck)
    return smooth_path[0], smooth_path[1]

def place_cones(x, y, num_cones):
    indices = np.linspace(0, len(x) - 1, num_cones).astype(int)
    cone_x = x[indices]
    cone_y = y[indices]
    return cone_x, cone_y

# Parameters
a_inner = 40  # Semi-major axis for inner ellipse
b_inner = 20   # Semi-minor axis for inner ellipse
a_outer = 43  # Semi-major axis for outer ellipse
b_outer = 23   # Semi-minor axis for outer ellipse
center = (0, 0)
num_cones = 80  # Total number of cones

# Create ellipse coordinates
inner_x, inner_y = create_ellipse(a_inner, b_inner, center)
outer_x, outer_y = create_ellipse(a_outer, b_outer, center)

# Smooth the ellipse path
smooth_inner_x, smooth_inner_y = smooth_path(inner_x, inner_y)
smooth_outer_x, smooth_outer_y = smooth_path(outer_x, outer_y)

# Place cones
inner_cone_x, inner_cone_y = place_cones(smooth_inner_x, smooth_inner_y, num_cones)
outer_cone_x, outer_cone_y = place_cones(smooth_outer_x, smooth_outer_y, num_cones)

# Plot the elliptical circuit with smooth edges
plt.figure(figsize=(8, 8))
plt.plot(smooth_inner_x, smooth_inner_y, 'b-', label='Inner Track')
plt.plot(smooth_outer_x, smooth_outer_y, 'r-', label='Outer Track')

# Adding inner and outer cones
plt.scatter(inner_cone_x, inner_cone_y, c='blue', marker='o', label='Inner Cones')
plt.scatter(outer_cone_x, outer_cone_y, c='red', marker='o', label='Outer Cones')

# Set equal scaling and add labels
plt.axis('equal')
plt.title('Elliptical Circuit with Smooth Edges and Cones')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
