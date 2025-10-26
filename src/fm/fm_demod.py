from typing import Callable, Any
from numpy.typing import NDArray
import numpy as np
from scipy.fft import fft,ifft
from scipy.signal import butter, lfilter, freqz
class fm_demod:
    def __init__(self,kf:float,fc:float,Ac:float,f_cutoff:float,fs:float,lpf_order:float):
        self.kf = kf
        self.fc = fc
        self.Ac = Ac
        self.fs = fs
        self.b,self.a = butter(N=lpf_order/2,Wn=f_cutoff*2/fs,btype='lowpass')
        
    def demodulate(self,psi:NDArray[np.float64]):
        del_psi = np.zeros(len(psi))
        for i in range(0,len(psi)):
            if i == 0:
                del_psi[i] = (psi[i+1] - psi[i])*self.fs # Linear interpolation at lower edge
            else:
                del_psi[i] = (psi[i] - psi[i-1])*self.fs # Backward difference
        print(del_psi)
        rect_del_psi = self.__rectify(del_psi)
        env = lfilter(self.b,self.a,rect_del_psi)
        env = -env/(2*np.pi*self.Ac)
        env = (env - self.fc)/self.kf
        return env
        
    def __rectify(self,x:NDArray[np.float64]):
        y = np.zeros(len(x))
        for i in range(0,len(x)):
            if x[i] > 0:
                y[i] = x[i]
        return y

