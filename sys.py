import numpy as np
import matplotlib.pyplot as plt
from fm_mod import fm_mod
'''
    Ac = 1
    Am = 1
    kv = 75
    fm = 10e3
    fc = 10e3

'''
# Sampling parameters
n:int = 2**16
fs:float = 1e7
Ts:float = 1/fs
t = np.arange(0,n*Ts,Ts)

# Message signal parameters
Am = 10
m = Am*np.cos(2*np.pi*1e3*t)

# Modulator parameters
kf = 75
Ac = 1
fc = 1e3

# Message signal plot
fig, ax = plt.subplots(nrows=2,ncols=1)
ax[0].plot(t,m)
ax[0].set(
    xlabel="Time (s)",
    ylabel="m(t)",
    title="Message Signal Waveform"
)

modulator:fm_mod = fm_mod(kf=kf,Ac=Ac,wc=2*np.pi*fc)
psi = modulator.modulate(m,t,Ts)
# print(psi)
# print(m)
ax[1].plot(t,psi)
ax[1].set(
    xlabel="Time (s)",
    ylabel="phi(t)",
    title="FM-Modulated Signal Waveform"
)
plt.show()


