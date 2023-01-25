import numpy as np
import itertools
import pandas as pd
import time

import masscompute
from masscompute import masseval
from featurize.msutil import mstools


f = {6: 20, 1: 20, 7: 4, 15: 5, 16: 3, 17: 4}

res = []

ITERS = 10
for i in range(ITERS):
    t1 = time.time()
    r = masseval.py_get_all_frag_spect_np_nostruct(f)
    t2 = time.time()
    r = masseval.py_get_all_frag_spect_highres(f)
    t3 = time.time()


    res.append({'orig_runtime' : t2-t1,
                'highres_runtime' : t3-t2})

df = pd.DataFrame(res)
print(df.mean())
