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
fs = 1e10
Ts = 1/fs
n = 2**18
t0 = 10/fm # To always depict steady-state behaviour
t = np.arange(start=t0,stop=t0+n*Ts,step=Ts)

# Channel noise params
psd = -180
sd = np.sqrt((10**(psd*fs/10))*1e-6)

# Message signal 
m = Am*np.cos(2*np.pi*fm*t)

# FM modulated signal
mod = fm_mod(kf=kf,fc=fc,Ac=Ac)
psi = mod.modulate(m=m,t=t,Ts=Ts)

# Additive channel noise
wgn = np.random.normal(loc=0,scale=sd,size=(1,n))

# Channel signal
x = psi + wgn

# FM demodulated signal
demod = fm_demod(kf=kf,fc=fc,Ac=Ac,f_cutoff=2*fm,fs=fs,lpf_order=1000)
m_noisy = demod.demodulate(x)
m_noiseless = demod.demodulate(psi)



