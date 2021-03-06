import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


g = 9.8  #in ms-2
L = 140  # DOMAIN SIZE in m 
ob_L = 4 # OBJECT SIZE in m
U = 0.4 * np.sqrt(g * ob_L)   # SPEED in ms-1
x_boat = 50 # Position of the boat on the x-axis (in mesh size)

N = 512  # POINTS
NT = 2000 # TIME
NB = 100 # NO POINTS IN ABSORBING BOUNDARY

dx = L / N
dt = dx / U
draw = 1

F = U / np.sqrt(g * ob_L)

# THEORETICAL SLOPE
if F < 0.49:
    slope = np.arctan(19.47 * np.pi / 180)
else:
    slope = np.sqrt(2 * np.pi * F ** 2 - 1) / (4 * np.pi * F ** 2 - 1)  

print("Fr: ", F)
print("dt: ", dt)

def phys_spec(UR, N):
    """Return the Fourier transform of UR."""
    Unew = np.fft.fftn(UR / N, (N, N))
    Unew = Unew[:int(N / 2), :int(N / 2)]

    return Unew

def spec_phys(U, N):
    """Reverse Fourier transform U."""
    Unew = N * np.fft.ifftn(U, (N, N))
    return Unew

eta_all = []
xtmp = np.linspace(-L / 2, L / 2, N + 1)

x = xtmp[0:N]
y = xtmp[0:N]
xx, yy = np.meshgrid(y, x)

eta = np.zeros([N, N])
#eta_l=np.exp(-100*((xx-np.pi)**2+(yy-np.pi)**2))
#eta_r=np.exp(-100*((xx-np.pi)**2+(yy-np.pi)**2))

eta_hat = phys_spec(eta, int(N / 2))

kx = np.linspace(0, np.pi * (N - 1) / L, int(N / 2))
ky = np.linspace(0, np.pi * (N - 1) / L, int(N / 2))

kxx, kyy = np.meshgrid(ky, kx)

omega = np.sqrt(g * np.sqrt(kxx ** 2 + kyy ** 2))

for it in range(NT):
    
    eta_o = eta.copy()
    eta = np.roll(eta, 1, 1)

    eta[:, -1:-NB] = -eta[:, -1:-NB] * (100 - dt)
    
    eta -= dt * np.exp(-2 * np.pi ** 2 * ((xx + x_boat) ** 2 + (yy) ** 2)/ ob_L **2 )\
        *(4 * np.pi ** 2 / ob_L ** 2) * (xx + x_boat) * 0.5 
    
    eta_hat = phys_spec(eta, N)

    eta_hat = eta_hat * np.exp(dt * omega *1j)

    eta = spec_phys(eta_hat, N )#*np.exp(-(xx/(L))**4)    
    
    # Visualisation
    if (it%10==0):
        

        eta_all.append(eta)
        print("{:10.2f}".format(it/NT))

        if draw==1:
            plt.figure(1)
            plt.clf()
            plt.pcolormesh(x, x, eta.real + np.flip(eta.real,0),\
                shading="auto", cmap="jet", vmin=-0.1, vmax=0.1)
            plt.title("Fr ={:10.2f}".format(F))
            plt.draw()
            plt.pause(0.001)

if draw==1:
    plt.plot(x, -slope * (x + x_boat), "--k")

plt.figure(1)
plt.pcolormesh(x, x, eta.real + np.flip(eta.real,0),\
     shading="auto", cmap="jet", vmin=-0.1, vmax=0.1)
