import numpy as np
import matplotlib.pyplot as plt
from fm.fm_mod import fm_mod
from fm.fm_demod import fm_demod
'''
    Ac = 1
    Am = 1
    kv = 75
    fm = 1e3
    fc = 1e6
'''
# Sampling parameters
n:int = 2**18
fs:float = 1e5
Ts:float = 1/fs
t = np.arange(0,n*Ts,Ts)

# Message signal parameters
Am = 1
fm = 10
m = Am*np.cos(2*np.pi*fm*t)

# Modulator parameters
kf = 3*1e3/4
Ac = 1
fc = 1e3

modulator:fm_mod = fm_mod(kf=kf,Ac=Ac,fc=fc)
demodulator:fm_demod = fm_demod(kf=kf,fc=fc,Ac=Ac,f_cutoff=fm/4,fs=fs,lpf_order=2000)
psi = modulator.modulate(m,t,Ts)
# print(psi)
demod = demodulator.demodulate(psi)

# Message signal plot
fig, ax = plt.subplots()

ax.plot(t,m,'r')
ax.plot(t,psi,'b')
ax.plot(t,demod,c='green')
ax.set(
    xlabel="Time (s)",
    ylabel="m(t)",
    title="Message Signal Waveform",
)
ax.set_xlim(left=1,right=1+2/fm)
ax.grid()
plt.show()
fig.savefig("docs/fm_demod_test.png")


