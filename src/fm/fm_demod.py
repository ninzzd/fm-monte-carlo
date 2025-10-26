from typing import Callable, Any
from numpy.typing import NDArray
import numpy as np
from scipy.fft import fft,ifft
from scipy.signal import butter, lfilter, freqz
from scipy import signal
class fm_demod:
    def __init__(self,kf:float,fc:float,Ac:float,f_cutoff:float,fs:float,lpf_order:float):
        self.kf = kf
        self.fc = fc
        self.Ac = Ac
        self.fs = fs
        # self.b,self.a = butter(N=lpf_order/2,Wn=f_cutoff*2/fs,btype='lowpass') # Non-linear phase is causing some drift in the demodulated signal
        self.b = signal.firwin(
            numtaps=lpf_order+1,
            cutoff=f_cutoff*2/fs,
            pass_zero='lowpass',
            window='hamming',
            fs=fs
        )
        
    def demodulate(self,psi:NDArray[np.float64]):
        diff_order = 1
        del_psi = np.diff(psi,n=diff_order,prepend=[psi[0] - (psi[1] - psi[0])])*self.fs/diff_order
        print(del_psi)
        rect_del_psi = self.__rectify(del_psi)
     
        # env = lfilter(self.b,self.a,rect_del_psi) # For butterworth filter
        a = np.zeros(len(self.b))
        a[0] = 1
        env = lfilter(b=self.b,a=a,x=rect_del_psi)
        m = env/(2*np.pi*self.Ac)
        m = (m - self.fc)/self.kf
        m = m - np.mean(m)
        return m
        
    def __rectify(self,x:NDArray[np.float64]):
        y = np.zeros(len(x))
        for i in range(0,len(x)):
            if x[i] > 0:
                y[i] = x[i]
        return y

