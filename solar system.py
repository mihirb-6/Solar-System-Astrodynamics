from math import sin, cos, log, exp, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

'''
Runge-Kutta-Felhberg Method
f: function to integreate over
a: starting point of interval
b: ending point " "
r0: initial val of state vector
tol: tolerance of accuracy at each step
rs and ts: values of r and t at each step, respectively
'''

def rkf45_curve(f,a,b,r0,tol):
    # Very crude initial guess for h
    h = (b-a)/float(100)
    t = a
    r = r0
    
    ts_list = []
    rs_list = []
    ts_list.append(t)
    rs_list.append(r)
    
    while True:
        # Assume we don't exit this step; change if needed
        break_now = False
        
        # Check whether current step size would carry past b, and
        # adjust h if so
        if t+h > b:
            h = b-t
            break_now = True
        # Perform the RK step
        k0 = h*f(t, r)
        k1 = h*f(t+h/4, r+k0/4)
        k2 = h*f(t + 3*h/8, r + 3*k0/32 + 9*k1/32)
        k3 = h*f(t + 12*h/13, r + 1932*k0/2197 - 7200*k1/2197 + 7296*k2/2197)
        k4 = h*f(t+h, r + 439*k0/216 - 8*k1 + 3680*k2/513 - 845/4104*k3)
        k5 = h*f(t+h/2, r - 8*k0/27 + 2*k1 - 3544*k2/2565 + 1859*k3/4104 - 11*k4/40)
        
        rstar = r + 25*k0/216 + 1408*k2/2565 + 2197*k3/4104 - k4/5
        rnew = r + 16*k0/135 + 6656*k2/12825 + 28561*k3/56430 - 9*k4/50 + 2*k5/55

        # Check the error; update t, r if allowed
        err = np.max(abs(rstar - rnew))
        if err < tol:
            t += h
            r = rstar
            ts_list.append(t)
            rs_list.append(r)

        # Exit if instructed to
        if break_now:
            break

        # Adjust step size
        if err != 0.0:
            h *= 0.8*(tol/err)**0.2
        else:
            h *= 5.0
    # Convert lists of ts, rs into numpy arrays
    ts = np.array(ts_list,float)
    rs = np.array(rs_list,float)
    
    return rs,ts

'''
if __name__ == '__main__':
    # Initial, final times
    t0 = 0
    tf = 1000
    
    # Initial conditions
    x0 = 0.12
    y0 = -0.92
    z0 = -0.40
    vx0 = 0.01448
    vy0 = 0.0002424
    vz0 = 0.0002815
    
    r0 = np.array([x0,y0,z0,vx0,vy0,vz0], float)
    
    # The required tolerance at each step
    tol = 1e-4#Set this
'''

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
r_data, t_data = rkf45_curve(grav, t0, tf, r0, tol)
# Use a dataframe to organize values
df = pd.DataFrame(r_data)
x = df[0]
y = df[1]
z = df[2]

# Testing two objects now:
r0_earth = np.array([1, 2, 3, 4, 5, 6], float)
r0_merc = np.array([6, 5, 4, 3, 2, 1], float)

r_earth_data, t_data = rkf45_curve(grav, t0, tf, r0_earth, tol)
r_merc_data, t_data = rkf45_curve(grav, t0, tf, r0_merc, tol)

df_E = pd.DataFrame(r_earth_data)
df_merc = pd.DataFrame(r_merc_data)


# Initialize plot and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3D plot of the positions
line, = ax.plot(df[0], df[1], df[2])
line, ax.plot(df_E[0], df_E[1], df_E[2])
line, ax.plot(df_merc[0], df_merc[1], df_merc[2])

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
#plt.show()
