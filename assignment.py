from fm.fm_mod import fm_mod
from fm.fm_demod import fm_demod
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

'''
    Ac = 1
    Am = 1
    kf = 5e5
    fm = 1e3
    fc = 1e8
    fs = 1e10
    n = 2**18
    psd : -180 dB/Hz : -170 dB/Hz
'''
# Message params
Am = 1
fm = 1e3

# Carrier params
Ac = 1
kf = 5e5
fc = 1e8

# Sampling params
fs = 50*fc
Ts = 1/fs
n = int(round(fs/fm))
t0 = 10/fm # To always depict steady-state behaviour
t = np.arange(start=0,stop=n*Ts,step=Ts)
print(t.shape)

# Channel noise params
psd = -180
print(psd*fs)
sd = np.sqrt((10**(psd*fs/10))*1e-6)
print(sd)

# Message signal 
m = Am*np.cos(2*np.pi*fm*t)

# FM modulated signal
mod = fm_mod(kf=kf,fc=fc,Ac=Ac)
psi = mod.modulate(m=m,t=t,Ts=Ts)

# Additive channel noise
wgn = np.random.normal(loc=0,scale=sd,size=(n,))
print(wgn.shape)

# Channel signal
x = psi + wgn

# FM demodulated signal
demod = fm_demod(kf=kf,fc=fc,Ac=Ac,f_cutoff=2*fm,fs=fs,lpf_order=1000)
m_noisy = demod.demodulate(x)
m_noiseless = demod.demodulate(psi)

# Plotting
fig, ax = plt.subplots()
ax.plot(t,m,'r')
#ax.plot(t,psi,'b')
ax.plot(t,m_noisy,'green')
#ax.plot(t,m_noiseless,'black')
ax.set(
    xlabel="Time (s)",
    ylabel="m(t)",
    title="Message Signal Waveform",
)
ax.set_ylim(ymin=-2.0,ymax=+2.0)
ax.grid()
plt.show()

