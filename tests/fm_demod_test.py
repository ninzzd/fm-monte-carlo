import numpy as np
import matplotlib.pyplot as plt
from fm.fm_mod import fm_mod
from fm.fm_demod import fm_demod
'''
    Ac = 1
    Am = 1
    kv = 750
    fm = 10
    fc = 1e3
    fs = 1e5
    n = 2**18
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
demodulator:fm_demod = fm_demod(kf=kf,fc=fc,Ac=Ac,f_uc=fm/4,fs=fs,lpf_order=1000)
psi = modulator.modulate(m,t,Ts)
# print(psi)
demod = demodulator.demodulate(psi)

# Plotting
fig, ax = plt.subplots()

ax.plot(t,m,'r')
ax.plot(t,psi,'b')
ax.plot(t,demod,c='g')
ax.set(
    xlabel="Time (s)",
    ylabel="m(t)",
    title="Message Signal Waveform",
)
t0 = 1
ax.set_xlim(left=t0,right=t0+2/fm)
ax.grid()
plt.show()
# fig.savefig("docs/fm_demod_test.png")


