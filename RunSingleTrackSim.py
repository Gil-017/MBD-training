"""
===========================
single track vehicle model simulation
===========================
"""


from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

# =================================
m=2295
L=3.0
mf=1170
mr=1125
a=L*mr/m
b=L*mf/m
Izz=4200
C1=3000*180/np.pi
C2=3500*180/np.pi   #3500*180/pi;
u=100/3.6   #120/3.6

# =================================

def EoM_singletrackvehicle(t,y):
    v = y[0]
    w = y[1]

    SR=16
    SWA=Steering_input(t)

    delta = SWA / SR / 180 * np.pi
    alpha_fr = delta - 1/u*(v+a*w)
    alpha_rr = -1/u*(v-b*w)

    Fy1=C1*alpha_fr
    Fy2=C2*alpha_rr

    ydot = np.zeros_like(y)

    ydot[0]=(Fy1+Fy2)/m - w*u
    ydot[1]=(a*Fy1-b*Fy2)/Izz

    print(ydot)

    return ydot

# ==============================

def Steering_input(t):
    if t<0.5:
        SWA=0.0
    elif t<0.7:
        SWA = 30 / 0.2 *(t-0.5)
    else:
        SWA = 30

    #SWA = 45 * sin(0.2*2*np.pi*t);

    return SWA


# ===============================
y0 = [0.0, 0.0]
t0=0.0
tf=3.0
tspan = np.arange(t0, tf, 0.01)
output = integrate.solve_ivp(EoM_singletrackvehicle, (t0, tf), y0, t_eval=tspan)

# ===============================
Beta = np.zeros_like(output.t)
swa=np.zeros_like(output.t)
ay=np.zeros_like(output.t)

for myindex in range(len(output.t)):
    swa[myindex] =  Steering_input(output.t[myindex])
    ydot = EoM_singletrackvehicle(output.t[myindex], [output.y[0,myindex], output.y[1,myindex]])
    ay[myindex] =  ydot[0] + output.y[1,myindex]*u
    Beta[myindex] = np.arctan2(output.y[0,myindex], u)

# ================================
plt.figure(1)

plt.subplot(2,2,1)
plt.plot(output.t, swa)
plt.ylabel('SWA [deg]')

plt.subplot(2,2,2)
plt.plot(output.t, Beta*180/np.pi)
plt.ylabel('Beta [deg]')

#plt.figure(2)
plt.subplot(2,2,3)
plt.plot(output.t, output.y[1]*180/np.pi)
plt.ylabel('Yaw rate [deg/s]')
plt.xlabel('time [s]')

plt.subplot(2,2,4)
plt.plot(output.t, ay/9.81)
plt.ylabel('Ay [g]')
plt.xlabel('time [s]')

plt.tight_layout()
plt.savefig('test.pdf')
#plt.show()
