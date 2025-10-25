import numpy as np
class fm_mod:
    def __init__(self,kf:float,wc:float,Ac:float):
        self.kf = kf
        self.wc = wc
        self.Ac = Ac
    def modulate(self,m:np.array,t:np.array,ts:float):
        phi = np.zeros(len(m))
        for i in range(0,len(m)):
            if i == 0:
                phi[i] = m[i]*ts
            else:
                phi[i] = phi[i-1] + m[i]*ts
        phi = phi*self.kf*2*np.pi
        print(phi)
        phi = phi + self.wc*t
        psi = self.Ac*np.cos(phi)
        return psi
