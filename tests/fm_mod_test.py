import numpy as np
import matplotlib.pyplot as plt
from fm.fm_mod import fm_mod
'''
    Ac = 1
    Am = 1
    kv = 75
    fm = 10e3
    fc = 10e3
'''
# Sampling parameters
n:int = 2**12
fs:float = 1e5
Ts:float = 1/fs
t = np.arange(0,n*Ts,Ts)

# Message signal parameters
Am = 1
fm = 10
m = Am*np.cos(2*np.pi*fm*t)

# Modulator parameters
kf = 2e3
Ac = 1
fc = 1e3

# Message signal plot
fig, ax = plt.subplots()

modulator:fm_mod = fm_mod(kf=kf,Ac=Ac,wc=2*np.pi*fc)
psi = modulator.modulate(m,t,Ts)

ax.plot(t,m,'r')
ax.plot(t,psi,'b')
ax.set(
    xlabel="Time (s)",
    ylabel="m(t)",
    title="Message Signal Waveform"
)
ax.grid()
plt.show()
fig.savefig("docs/fm_mod_test.png")


