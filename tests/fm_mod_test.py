import numpy as np
import matplotlib.pyplot as plt
from fm.fm_mod import fm_mod
'''
    Ac = 1
    Am = 1
    kf = 750
    fm = 10
    fc = 1e3
    fs = 1e5
    n = 2**16
'''
# Sampling parameters
n:int = 2**16
fs:float = 1e5
Ts:float = 1/fs
t = np.arange(0,n*Ts,Ts)
print(len(t))

# Message signal parameters
Am = 1
fm = 10
m = Am*np.cos(2*np.pi*fm*t)

# Modulator parameters
kf = 3*1e3/4
Ac = 1
fc = 1e3

# Message signal plot
fig, ax = plt.subplots()

modulator:fm_mod = fm_mod(kf=kf,Ac=Ac,fc=fc)
psi = modulator.modulate(m,t,Ts)

ax.plot(t,m,'r')
ax.plot(t,psi,'b')
ax.set(
    xlabel="Time (s)",
    ylabel="m(t)",
    title="Message Signal Waveform"
)
t0 = 0.3
ax.set_xlim(left=t0,right=t0+2/fm)
ax.grid()
plt.show()
# fig.savefig("docs/fm_mod_test.png")


