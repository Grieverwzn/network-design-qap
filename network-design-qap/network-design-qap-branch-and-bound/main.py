from paramsQAP import ParamsQAP
from glb import GLB
from branchandbound4 import BAB
import time
import numpy as np

#cwd = r'..\large_scale_synchronization'
cwd = r'..\Had_12'
args = {'node_id_unit': 1000,
        'M': 1000000,
        'step_size': 0.5,
        'target_relative_gap': 0.01,
        'max_branch_iters': 100,
        'threads': 7,                   # -1: max threads, n>0: n threads
        'time_limit': 360          # seconds
        }
init_UB=np.inf
#init_UB=24170

if __name__ == '__main__':
    instance = ParamsQAP(cwd, args)
    time_start=time.time()
    glbsolver = GLB(init_UB)
    glbsolver.calculateGLB(instance)
    bab = BAB(instance, glbsolver, args, cwd)
    bab.solve()
    time_end=time.time()
    print('TotalCPU Time:',time_end-time_start,'s...')

