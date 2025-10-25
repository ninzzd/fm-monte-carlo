import numpy as np
import matplotlib.pyplot as plt
'''
    Ac = 1
    Am = 1
    kv = 75
    fm = 10e3
    fc = 10e3

'''
# Sampling parameters
n:int = 4096
fs:float = 1e6
Ts:float = 1/fs
t = np.arange(0,n*Ts,Ts)

# Message signal parameters
Ac = 1
m = Ac*np.cos(2*np.pi*1e3*t)

# Message signal plot
fig, ax = plt.subplots()
ax.plot(t,m)
ax.set(
    xlabel="Time (s)",
    ylabel="m(t)",
    title="Message Signal Waveform"
)
ax.grid()
plt.show()