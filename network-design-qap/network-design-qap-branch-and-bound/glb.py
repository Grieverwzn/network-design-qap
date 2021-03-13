import time
import numpy as np
from hungarian import *


class GLB:
    def __init__(self,init_UB):
        self.LB = -1*np.inf
        self.UB = init_UB
        self.best_solution_1 = None
        self.best_solution_2 = None

    def calculateGLB(self, instance):
        print('Calculate GLB...')
        t0 = time.time()
        g_assignment_mat_1 = np.zeros([instance.nb_of_orig_building, instance.nb_of_orig_location])
        g_assignment_mat_2 = np.zeros([instance.nb_of_dest_location, instance.nb_of_dest_building])
        # [0] GLB sub problem
        lower_solution_1 = Hungarian_1(instance.GLB_cost_mat + instance.build_cost_orig_mat)
        # assignment_count_mat = Trace(lower_solution_1)
        for i in range(instance.nb_of_orig_location):
            k_ind = lower_solution_1['location_ind'][i]
            i_ind = lower_solution_1['building_ind'][i]
            g_assignment_mat_1[i_ind][k_ind] = 1

        lower_solution_2 = Hungarian_2(instance.build_cost_dest_mat)
        for i in range(instance.nb_of_dest_location):
            l_ind = lower_solution_2['location_ind'][i]
            j_ind = lower_solution_2['building_ind'][i]
            g_assignment_mat_2[l_ind][j_ind] = 1

        # print('the lead assignment has ',lower_solution_1['value'])
        # print('the follower assignment has ', lower_solution_2['value'])
        LowerBound = lower_solution_1['value'] + lower_solution_2['value']
        # g_lower_bound_list.append(LowerBound)

        print('  calculate upper bound')
        UpperBound = np.sum(instance.flow_mat * np.matmul(np.matmul(g_assignment_mat_1, instance.trans_cost_mat), g_assignment_mat_2)) + \
                     np.sum(instance.build_cost_orig_mat * g_assignment_mat_1) + np.sum(instance.build_cost_dest_mat * g_assignment_mat_2)


        if LowerBound >= self.LB:
            self.LB = LowerBound
        if UpperBound <= self.UB:
            self.UB = UpperBound
            self.best_solution_1 = g_assignment_mat_1
            self.best_solution_2 = g_assignment_mat_2
        GAP = np.abs(self.UB - self.LB) / self.UB
        print('  GLB - Lower bound = ', self.LB)
        print('  GLB - Upper bound = ', self.UB)
        print('  GAP = ', GAP)

        t1 = time.time()
        total_time = round(t1-t0, 2)
        print(f'  time used: {total_time}s')

