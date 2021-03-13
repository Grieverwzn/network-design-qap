from paramsQAP import ParamsQAP
from branchandbound4 import BAB


# cwd = r'..\test_demo'
# cwd = r'..\QAP_Data_Nug_12'
cwd = r'..\large_scale_synchronization_r4'

args = {'node_id_unit': 1000,
        'M': 1000000,
        'target_relative_gap': 1e-3,
        'threads': 7,                       # -1: max threads, n>0: n threads
        'time_limit': 360                  # seconds
        }


if __name__ == '__main__':
    instance = ParamsQAP(cwd, args)
    bab = BAB(instance, args)
    bab.solve()

