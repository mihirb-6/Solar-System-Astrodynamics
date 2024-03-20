import gravitational_force
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Function
def grav(t, r):
    mu = 1
    x = r[0]
    y = r[1]
    z =  r[2]
    
    xdot = r[3]
    ydot = r[4]
    zdot = r[5]
    dx = xdot
    dy = ydot
    dz = zdot
    
    dxdot = -(mu*x) / ((x**2+y**2+z**2)**(3/2))
    dydot = -(mu*y) / ((x**2+y**2+z**2)**(3/2))
    dzdot = -(mu*z) / ((x**2+y**2+z**2)**(3/2))
    
    rdot = np.array([dx,dy,dz,dxdot,dydot,dzdot],float)
    
    return rdot


# Initial, Final Times
t0 = 1
tf = 1e5

# Initial positions, velocities
x0 = -18.26
y0 = -1.16
z0 = -0.25
vx0 = 0.02
vy0 = -0.001
vz0 = -0.006

# Positions and Velocities stored in an array
r0 = np.array([x0,y0,z0,vx0,vy0,vz0],float)

#Set the desired tolerance
tol = 1e-8

# Call force function
r_data, t_data = gravitational_force.rkf45_curve(grav, t0, tf, r0, tol)

# Use a dataframe to organize values
df = pd.DataFrame(r_data)
x = df[0]
y = df[1]
z = df[2]

# Initialize plot and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3D plot of the positions
line, = ax.plot(df[0], df[1], df[2])

# Setting the limits for the axes
ax.set_xlim(min(df[0]), max(df[0]))
ax.set_ylim(min(df[1]), max(df[1]))
ax.set_zlim(min(df[2]), max(df[2]))

# The function that will update the plot at each frame
def update_data(frame, data, line):
    line.set_data(data[0][:frame], data[1][:frame])
    line.set_3d_properties(data[2][:frame])
    return line

# Defining parameters for the actual animation
animation = FuncAnimation(fig,
                          update_data,
                          frames=len(df),
                          fargs=(df, line),
                          interval=50,
                          repeat=False,
                          blit=False)

# Show the animation
plt.show()
