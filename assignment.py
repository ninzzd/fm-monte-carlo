from fm.fm_mod import fm_mod
from fm.fm_demod import fm_demod
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

'''
    Ac = 1
    Am = 1
    kv = 750
    fm = 10
    fc = 1e3
    fs = 1e5
    n = 2**18
    psd : 0 dB/Hz - 30 dB/Hz
'''

# Channel noise characteristics
