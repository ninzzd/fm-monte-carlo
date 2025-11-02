from fm.fm_mod import fm_mod
from fm.fm_demod import fm_demod
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib
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

# ---------------------------------------------------------------------------------

# Message params
Am = 1
fm = 1e3

# ---------------------------------------------------------------------------------

# Carrier params
Ac = 1
kf = 5e5
fc = 1e8

# ---------------------------------------------------------------------------------

# Sampling params
fs = 10*fc
Ts = 1/fs
n = int(round(8*fs/fm))
t0 = 10/fm # To always depict steady-state behaviour
t = np.arange(start=0,stop=n*Ts,step=Ts)
# print(t.shape)
t1 = 0.004 # Steady-state start and end points based on graphical observation
t2 = 0.006

# ---------------------------------------------------------------------------------

# Channel noise params
psd = -180
sd = np.sqrt((10**(psd/10))*1e-6*fs)
# print(sd)

# ---------------------------------------------------------------------------------

# Message signal 
m = Am*np.cos(2*np.pi*fm*t)

# ---------------------------------------------------------------------------------

# FM modulated signal
mod = fm_mod(kf=kf,fc=fc,Ac=Ac)
psi = mod.modulate(m=m,t=t,Ts=Ts)

# ---------------------------------------------------------------------------------

# Additive channel noise
wgn = np.random.normal(loc=0,scale=sd,size=(n,))
# print(wgn.shape)

# ---------------------------------------------------------------------------------

# Channel signal
x = psi + wgn

# ---------------------------------------------------------------------------------

# FM demodulated signal
demod = fm_demod(kf=kf,fc=fc,Ac=Ac,f_lc=0.5*fm,f_uc=2*fm,fs=fs,order=4,gain=5)
m_noisy = demod.demodulate(x)
# m_noiseless = demod.demodulate(psi)
# print(m_noisy)

# ---------------------------------------------------------------------------------

# Steady-state samples
n1 = int(np.floor(t1*fs))
n2 = int(np.floor(t2*fs))
n_ss = n2-n1
t_ss = t[n1:n2] # Time samples
m_ss = m[n1:n2] # Message signal
m_noisy_ss = m_noisy[n1:n2] # Noisy demodulated signal
psi_ss = psi[n1:n2] # Noiseless modulated/channel signal
x_ss = x[n1:n2] # Noisy modulated/channel signal

# ---------------------------------------------------------------------------------

# Power calculation
p_m_ss = np.sum(m_ss**2)/n_ss
p_err_ss = np.sum(m_ss**2 - m_noisy_ss**2)/n_ss
snr_out = p_m_ss/p_err_ss
p_psi_ss = np.sum(psi_ss**2)/n_ss
snr_in = p_psi_ss/sd**2
fom = snr_out/snr_in
print(f'Figure of Merit for {psd} dB/Hz = {fom}')
# ---------------------------------------------------------------------------------
'''
# FFT
t_ss = t[int(np.floor(t1*fs)):int(np.floor(t2*fs))]
m_ss = m[int(np.floor(t1*fs)):int(np.floor(t2*fs))]
m_noisy_ss = m_noisy[int(np.floor(t1*fs)):int(np.floor(t2*fs))]
print(f'length of m_ss={len(m_ss)}')
print(f'length of m_noisy_ss={len(m_noisy_ss)}')
ad_pad = 2 # Additional padding for increasing FFT resolution
n_ = int(2**(ad_pad+np.ceil(np.log2(len(m_ss)))))
m_padded = np.pad(
    array=m_ss,
    pad_width=(0,n_-len(m_ss))
)
m_noisy_padded = np.pad(
    array=m_noisy_ss,
    pad_width=(0,n_-len(m_ss))
)
# print(f'length of m_padded={len(m_padded)}')
# print(f'length of m_noisy_padded={len(m_noisy_padded)}')
m_fft = scipy.fft.fftshift(scipy.fft.fft(m_padded))
m_noisy_fft = scipy.fft.fftshift(scipy.fft.fft(m_noisy_padded))
'''
# ---------------------------------------------------------------------------------

# Plotting
# Time-domain waveform

fig, ax = plt.subplots(nrows=1,ncols=1)
ax.plot(t_ss,m_ss,'r',label='Message Signal')
#ax[0].plot(t,psi,'b',ls=':',alpha=0.2,label='FM Modulated Signal')
ax.plot(t_ss,m_noisy_ss,'green',label='Demodulated Signal')
#ax.plot(t,m_noiseless,'black')
ax.set(
    xlabel="Time (s)",
    ylabel="m(t)",
    title="Message and Demodulation Signal Waveform",
)
ax.set_ylim(
    ymin=-3.0,
    ymax=+2.0
)
ax.set_xlim(
    xmin=t1,
    xmax=t2
)
ax.grid()
plt.show()
'''
#Frequency spectrum
w = np.arange(
    start = -fs/2,
    step = fs/n_,
    stop = fs/2
)
ax[1].plot(w,np.abs(m_fft),'r',label='FFT of Message Signal')
#ax.plot(t,psi,'b')
ax[1].plot(w,np.abs(m_noisy_fft),'green',label='FFT of Demodulated Signal')
ax[1].legend()
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
'''
