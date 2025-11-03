from fm.fm_mod import fm_mod
from fm.fm_demod import fm_demod
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib
import scipy.signal
import scipy.fft 
import time

'''
    Ac = 1
    Am = 1
    kf = 5e5
    fm = 1e3
    fc = 1e8
    fs = 1e10
    psd : -180 dB/Hz : -170 dB/Hz
'''
exec_start = time.perf_counter()
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

# Message signal 
m = Am*np.cos(2*np.pi*fm*t)

# ---------------------------------------------------------------------------------

# FM modulated signal
mod = fm_mod(kf=kf,fc=fc,Ac=Ac)
psi = mod.modulate(m=m,t=t,Ts=Ts)

# ---------------------------------------------------------------------------------

psd_arr = np.arange(
    start=-180,
    stop=-169,
    step=1
)
fom_arr = np.zeros(len(psd_arr))
print(psd_arr)
print(fom_arr)
# ---------------------------------------------------------------------------------

# Monte-Carlo simulation for FoM vs PSD
n_mc = 75
j = 0
for psd in psd_arr:
    fom_sum = 0
    for i in range(0,n_mc):

        # ---------------------------------------------------------------------------------

        # Channel noise params
        R = 50 # Channel characteristic impedence
        sd = np.sqrt((10**(psd/10))*1e-3*R*fs)
        # print(sd)

        # ---------------------------------------------------------------------------------

        # Additive channel noise
        wgn = np.random.normal(loc=0,scale=sd,size=(n,))
        # print(wgn.shape)

        # ---------------------------------------------------------------------------------

        # Channel signal
        x = psi + wgn

        # ---------------------------------------------------------------------------------

        # FM demodulated signal
        demod = fm_demod(kf=kf,fc=fc,Ac=Ac,f_lc=0.5*fm,f_uc=2*fm,fs=fs,order=4,gain=1)
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
        m_noisy_ss = m_noisy_ss - np.mean(m_noisy_ss)
        rms_m = np.sqrt(np.mean(m_ss**2))
        rms_m_noisy_ss = np.sqrt(np.mean(m_noisy_ss**2))
        amp_gain = rms_m/rms_m_noisy_ss # RMS ratio based amplification
        m_noisy_ss = amp_gain*m_noisy_ss
        psi_ss = psi[n1:n2] # Noiseless modulated/channel signal
        x_ss = x[n1:n2] # Noisy modulated/channel signal

        # ---------------------------------------------------------------------------------

        # Power calculation
        p_m_ss = np.sum(m_ss**2)/n_ss
        p_err_ss = np.sum((m_ss - m_noisy_ss)**2)/n_ss
        p_psi_ss = np.sum(psi_ss**2)/n_ss
        var_n = sd**2
        snr_out = p_m_ss/p_err_ss
        snr_in = p_psi_ss/var_n
        fom = snr_out/snr_in
        fom_sum += fom
        curr_time = time.perf_counter()
        diff_time = curr_time - exec_start
        print(f'For {psd} dB/Hz: Exp {i+1}) Figure of Merit  = {fom} [Elapsed time: {diff_time*1e3} ms]:')
        print(f'\tMessage signal power={p_m_ss}')
        print(f'\tError signal power={p_err_ss}')
        print(f'\tModulated signal power={p_psi_ss}')
        print(f'\tNoise power={var_n}')

        # ---------------------------------------------------------------------------------

    fom_arr[j] = fom_sum/n_mc
    j += 1
# ---------------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(psd_arr,fom_arr,c='red')
ax.set_title(rf'Monte-Carlo Simulation for Figure of Merit vs Power Spectral Density ($N_{{iter}}={n_mc}$)')
ax.set_xlabel(r'Power Spectral Density ($dB/Hz$)')
ax.set_ylabel(f'Figure of Merit')
ax.grid()
plt.show()
fig.savefig(fname=rf'docs/monte_carlo_FoMvPSD_{n_mc}.png',format='png')