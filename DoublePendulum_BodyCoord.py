"""
===========================
The double pendulum problem
===========================

This animation illustrates the double pendulum problem.
"""

# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

# global variables
m1 = 1
m2 = 1
J1 = 1
J2 = 1
L1 = 1
L2 = 1
g = 9.81
K = 0.0



# =======================================
def diff_s(t, y):

    M = mass_matrix(y)
    h = force_vector(y)
    gamma = rhs_vector(y)
    D = jacobian_matrix(y)

    DMinv = np.matmul(D, np.linalg.inv(M))
    DMinvDt = np.matmul(DMinv, D.T)  #*D.T #D * np.linalg.solve(M, D.T)

    tmp = gamma - np.matmul(D, np.linalg.solve(M,h))
    L = np.linalg.solve(DMinvDt, tmp)
    acc = np.linalg.solve(M, h+np.matmul(D.T,L))

    ydot = np.append(y[6:12], acc)
    return ydot

# =======================================
def jacobian_matrix(y):
    r_1 = [1,0, -L1/2*np.cos(y[2]), 0,0, 0]
    r_2 = [0, 1, -L1/2*np.sin(y[2]), 0,0,0]
    r_3 = [-1, 0, -L1/2*np.cos(y[2]), 1 , 0, -L2/2*np.cos(y[5])]
    r_4 = [0,-1, -L1/2*np.sin(y[2]), 0, 1 , -L2/2*np.sin(y[5])]

    D = np.array([r_1, r_2, r_3, r_4])

    return D

# ======================================
def rhs_vector(y):

    v_1 = L1/2*np.sin(y[2])*y[8]**2
    v_2 = -L1/2*np.cos(y[2])*y[8]**2
    v_3 = L1/2*np.sin(y[2])*y[8]**2 + L2/2*np.sin(y[5])*y[11]**2
    v_4 = -L1/2*np.cos(y[2])*y[8]**2 - L2/2*np.cos(y[5])*y[11]**2

    gamma = - np.array([v_1,v_2,v_3,v_4])

    return gamma

# ======================================
def force_vector(y):

    h_1 = 0
    h_2 = -m1*g
    h_3 = K*(y[5]-y[2])
    h_4 = 0
    h_5 = -m2*g
    h_6 = -K*(y[5]-y[2])

    h = np.array([h_1,h_2,h_3,h_4,h_5,h_6])
    return h

# ======================================
def mass_matrix(y):

    M = np.diag([m1,m1,J1,m2,m2,J2])

    return M

# =====================================
# create a time array from 0..100 sampled at 0.01 second steps
dt = 0.01
t = np.arange(0.0, 20, dt)

# state varilables
r1x = L1/2
r1y = 0
r1th = np.pi/2
r2x = L1
r2y = -L2/2
r2th = 0

# initial state
state = np.array([r1x,r1y,r1th,r2x,r2y,r2th,0,0,0,0,0,0])

# integrate your ODE using scipy.integrate.
output = integrate.solve_ivp(diff_s, (0,20), state, t_eval = t, method='RK45',
                  rtol=1e-3, atol=1e-6)

# print(output.y)
# print(output.t)

y = output.y
t = output.t

plt.figure()
plt.plot(t,y[2])
plt.plot(t,y[5])

# x1 = L1*sin(y[0])
# y1 = -L1*cos(y[0])
#
# x2 = L2*sin(y[2]) + x1
# y2 = -L2*cos(y[2]) + y1
#
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()
p1x = y[0] - L1/2*np.sin(y[2])
p1y = y[1] + L1/2*np.cos(y[2])
p2x = y[0] + L1/2*np.sin(y[2])
p2y = y[1] - L1/2*np.cos(y[2])

p3x = y[3] - L2/2*np.sin(y[5])
p3y = y[4] + L2/2*np.cos(y[5])
p4x = y[3] + L2/2*np.sin(y[5])
p4y = y[4] - L2/2*np.cos(y[5])
#
line1, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    return line1, line2, time_text


def animate(i):
    body1_x = [p1x[i], p2x[i]]
    body1_y = [p1y[i], p2y[i]]
    body2_x = [p3x[i], p4x[i]]
    body2_y = [p3y[i], p4y[i]]

    line1.set_data(body1_x, body1_y)
    line2.set_data(body2_x, body2_y)
    time_text.set_text(time_template % (i*dt))
    return line1, line2, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y[0])),
                              interval=25, blit=True, init_func=init)

#ani.save('double_pendulum.mp4', fps=15)
plt.show()
