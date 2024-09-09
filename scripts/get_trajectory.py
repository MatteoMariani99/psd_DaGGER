import matplotlib.pyplot as plt


# Caricamento delle traiettorie da file: TRAIETTORIE CONTROLLORE
file_path_1 = "trajectory_poses/c_pose_cones_track9.txt"
x_data_1 = []
y_data_1 = []

with open(file_path_1, "r") as f1:
    for line in f1:
        x, y = map(float, line.strip().split(","))
        x_data_1.append(x)
        y_data_1.append(y)


# Caricamento delle traiettorie da file: TRAIETTORIE AGENTE
file_path_2 = "trajectory_poses/pose_cones_track9.txt"
x_data_2 = []
y_data_2 = []

with open(file_path_2, "r") as f2:
    for line in f2:
        x, y = map(float, line.strip().split(","))
        x_data_2.append(x)
        y_data_2.append(y)


# PLOT DI ENTRAMBE LE TRAIETTORIE
plt.figure()
plt.plot(x_data_1, y_data_1, 'b-', label='Controller trajectory')  # controllore in blu
plt.plot(x_data_2, y_data_2, 'r-', label='Agent trajectory')  # agente in rosso
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Overlapping Trajectories')
plt.legend()
plt.grid(True)

# Per mostrare il plot
plt.show()

