import numpy as np
class fm_mod:
    def __init__(self,kf:float,fc:float,Ac:float):
        self.kf = kf
        self.fc = fc
        self.Ac = Ac
    def modulate(self,m:np.array,t:np.array,Ts:float):
        phi = np.zeros(len(m))
        for i in range(0,len(m)):
            if i == 0:
                phi[i] = m[i]*Ts
            else:
                phi[i] = phi[i-1] + m[i]*Ts
        phi = phi*self.kf*2*np.pi
        phi = phi + 2*np.pi*self.fc*t
        psi = self.Ac*np.cos(phi)
        return psi
