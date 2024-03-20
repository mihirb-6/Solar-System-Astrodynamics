import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Function to update the plot with new data
def update_frame(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])
    return line

# Create a figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate some random data
data = np.random.rand(3, 100)

# Plot the data
line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

# Set limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Animate the plot
ani = FuncAnimation(fig, update_frame, frames=data.shape[1], fargs=(data, line), interval=50)

plt.show()
