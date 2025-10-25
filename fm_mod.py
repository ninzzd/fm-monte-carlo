import numpy as np
class fm_mod:
    def __init__(self,kf:float,wc:float,Ac:float):
        self.kf = kf
        self.wc = wc
        self.Ac = Ac
    def modulate(self,m:np.array,t:np.array):
        del_t:float
        if len(t) > 1:
            del_t = t[1] - t[0] # uniform time samples (obviously)
        psi = np.zeros()
