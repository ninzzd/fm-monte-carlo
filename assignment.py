from fm.fm_mod import fm_mod
from fm.fm_demod import fm_demod
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.fft 

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
fs = 25*fc
Ts = 1/fs
n = int(round(8*fs/fm))
t0 = 10/fm # To always depict steady-state behaviour
t = np.arange(start=0,stop=n*Ts,step=Ts)
print(t.shape)

# Channel noise params
psd = -180
sd = np.sqrt((10**(psd/10))*1e-6*fs)
print(sd)

# Message signal 
m = Am*np.cos(2*np.pi*fm*t)

# FM modulated signal
mod = fm_mod(kf=kf,fc=fc,Ac=Ac)
psi = mod.modulate(m=m,t=t,Ts=Ts)

# Additive channel noise
wgn = np.random.normal(loc=0,scale=sd,size=(n,))
# print(wgn.shape)

# Channel signal
x = psi + wgn

# FM demodulated signal
demod = fm_demod(kf=kf,fc=fc,Ac=Ac,f_cutoff=2*fm,fs=fs,lpf_order=4)
m_noisy = demod.demodulate(x)
m_noiseless = demod.demodulate(psi)
print(m_noisy)

# FFT
n_ = int(2**(np.ceil(np.log2(n))))
m_padded = np.pad(
    array=m,
    pad_width=(0,n_-n)
)
m_noisy_padded = np.pad(
    array=m_noisy,
    pad_width=(0,n_-n)
)
m_fft = scipy.fft.fftshift(scipy.fft.fft(m_padded))
m_noisy_fft = scipy.fft.fftshift(scipy.fft.fft(m_noisy_padded))

# Plotting
# Time-domain waveform
fig, ax = plt.subplots(nrows=2,ncols=1)
ax[0].plot(t,m,'r')
#ax.plot(t,psi,'b')
ax[0].plot(t,m_noisy,'green')
#ax.plot(t,m_noiseless,'black')
ax[0].set(
    xlabel="Time (s)",
    ylabel="m(t)",
    title="Message and Demodulation Signal Waveform",
)
ax[0].set_ylim(ymin=-2.0,ymax=+3.0)
ax[0].set_xlim(
    left=2/fm,
    right=3/fm
)
ax[0].grid()
#Frequency spectrum
w = np.arange(
    start = -fs/2,
    step = fs/n_,
    stop = fs/2
)
ax[1].plot(w,np.abs(m_fft),'r')
#ax.plot(t,psi,'b')
ax[1].plot(w,np.abs(m_noisy_fft),'green')
#ax.plot(t,m_noiseless,'black')
ax[1].set(
    xlabel=r'Frequency ($\omega$)',
    ylabel=r'M($\omega$)',
    title="Message and Demodulation Frequency Spectrum",
)
ax[1].set_xlim(
    left=-2*fm,
    right=+2*fm
)
ax[1].grid()

plt.show()

